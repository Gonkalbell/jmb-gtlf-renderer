use super::bind_groups;

use std::iter;
use std::{
    borrow::Cow,
    fmt::Write,
    sync::{Arc, Mutex},
};

use glam::{Mat4, Quat, Vec3};
use image::DynamicImage;
use reqwest::Url;
use wgpu::util::DeviceExt;

#[derive(Clone, Debug)]
pub struct Asset {
    pub info: String,
    pub tlas_bgroup: bind_groups::AccStructure,
}

pub struct LoadingProgress {
    pub loaded: usize,
    pub total: usize,
}

pub async fn load_asset(
    url: Url,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    loading_progress: Arc<Mutex<LoadingProgress>>,
) -> anyhow::Result<Asset> {
    if let Ok(mut loading_progress) = loading_progress.lock() {
        loading_progress.loaded = 0;
        loading_progress.total = 1;
    }
    let gltf_file = gltf::Gltf::from_slice(&request_data(&url, loading_progress.clone()).await?)?;
    let doc = gltf_file.document;

    let mut buffer_contents = futures::future::try_join_all(doc.buffers().map(|doc_buffer| {
        use gltf::buffer::Source;
        let url = url.clone();
        let source = doc_buffer.source().clone();
        let loading_progress = loading_progress.clone();
        async move {
            let contents = match source {
                Source::Bin => Vec::new(),
                Source::Uri(uri) => request_data(&url.join(uri)?, loading_progress).await?,
            };
            Ok::<_, anyhow::Error>(contents)
        }
    }))
    .await?;

    if let Some(blob) = gltf_file.blob {
        buffer_contents[0] = blob
    }

    let mut buffers: Vec<_> = doc
        .views()
        .map(|doc_view: gltf::buffer::View| {
            let start = doc_view.offset();
            let end = doc_view.offset() + doc_view.length();
            let contents = &buffer_contents[doc_view.buffer().index()][start..end];
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: doc_view.name(),
                contents,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::BLAS_INPUT,
            })
        })
        .collect();

    fixup_u8_index_buffers(&doc, &buffer_contents, &mut buffers, device);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("build acceleration structures"),
    });

    let textures = generate_textures(
        &url,
        device,
        queue,
        &mut encoder,
        &loading_progress,
        &doc,
        &buffer_contents,
    )
    .await?;

    let samplers: Vec<_> = doc
        .samplers()
        .map(|doc_sampler| {
            use gltf::texture::{MagFilter, MinFilter, WrappingMode};

            device.create_sampler(&wgpu::SamplerDescriptor {
                label: doc_sampler.name(),
                address_mode_u: match doc_sampler.wrap_s() {
                    WrappingMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
                    WrappingMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
                    WrappingMode::Repeat => wgpu::AddressMode::Repeat,
                },
                address_mode_v: match doc_sampler.wrap_t() {
                    WrappingMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
                    WrappingMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
                    WrappingMode::Repeat => wgpu::AddressMode::Repeat,
                },
                mag_filter: match doc_sampler.mag_filter() {
                    Some(MagFilter::Nearest) => wgpu::FilterMode::Nearest,
                    Some(MagFilter::Linear) | None => wgpu::FilterMode::Linear,
                },
                min_filter: match doc_sampler.min_filter() {
                    Some(
                        MinFilter::Nearest
                        | MinFilter::NearestMipmapLinear
                        | MinFilter::NearestMipmapNearest,
                    ) => wgpu::FilterMode::Nearest,
                    None
                    | Some(
                        MinFilter::Linear
                        | MinFilter::LinearMipmapLinear
                        | MinFilter::LinearMipmapNearest,
                    ) => wgpu::FilterMode::Linear,
                },
                mipmap_filter: match doc_sampler.min_filter() {
                    Some(
                        MinFilter::Nearest
                        | MinFilter::LinearMipmapNearest
                        | MinFilter::NearestMipmapNearest,
                    ) => wgpu::MipmapFilterMode::Nearest,
                    _ => wgpu::MipmapFilterMode::Linear,
                },
                ..Default::default()
            })
        })
        .collect();

    let default_texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("default texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
        },
        Default::default(),
        &[0xFF, 0xFF, 0xFF, 0xFF],
    );
    let default_sampler = device.create_sampler(&Default::default());

    let mesh_blases = generate_meshes(device, &mut encoder, &doc, &buffers);

    let tlas = generate_tlas(device, &mut encoder, &doc, &mesh_blases);

    let tlas_bgroup = bind_groups::AccStructure::from_bindings(
        device,
        bind_groups::AccStructureLayout { acc_struct: &tlas },
    );

    queue.submit(iter::once(encoder.finish()));

    let mut asset_info = String::new();
    let json_asset = &doc.as_json().asset;
    writeln!(&mut asset_info, "version: {}", json_asset.version)?;
    if let Some(min_version) = &json_asset.min_version {
        writeln!(&mut asset_info, "min_version: {}", min_version)?;
    }
    if let Some(copyright) = &json_asset.copyright {
        writeln!(&mut asset_info, "copyright: {}", copyright)?;
    }
    if let Some(generator) = &json_asset.generator {
        writeln!(&mut asset_info, "generator: {}", generator)?;
    }

    if let Ok(mut loading_progress) = loading_progress.lock() {
        loading_progress.loaded = loading_progress.total;
    }

    log::info!("finished loading {}", &url);
    Ok(Asset {
        info: asset_info,
        tlas_bgroup,
    })
}

// wgpu does not allow index buffers to be u8s, so I create new u16 index buffers for them.
fn fixup_u8_index_buffers(
    doc: &gltf::Document,
    buffer_contents: &[Vec<u8>],
    buffers: &mut [wgpu::Buffer],
    device: &wgpu::Device,
) {
    for accessor in doc.accessors() {
        use gltf::{accessor::DataType, buffer::Target};
        if let Some(view) = accessor.view()
            && let Some(Target::ElementArrayBuffer) = view.target()
            && accessor.data_type() == DataType::U8
        {
            let start = view.offset();
            let end = start + view.length();
            let u8_contents = &buffer_contents[view.buffer().index()][start..end];
            let u16_contents: Vec<u16> = u8_contents.iter().map(|&b| b as u16).collect();

            let u16_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: accessor.name(),
                contents: bytemuck::cast_slice(&u16_contents),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::BLAS_INPUT,
            });

            buffers[view.index()] = u16_index_buffer;
        }
    }
}

async fn request_data(
    url: &Url,
    loading_progress: Arc<Mutex<LoadingProgress>>,
) -> anyhow::Result<Vec<u8>> {
    if let Ok(mut loading_progress) = loading_progress.lock() {
        loading_progress.total += 1;
    }
    let data = match url.scheme() {
        "http" | "https" => reqwest::get(url.clone()).await?.bytes().await?.to_vec(),
        #[cfg(not(target_family = "wasm"))]
        "file" => {
            let path = url
                .to_file_path()
                .map_err(|_| anyhow::anyhow!("Invalid file URL"))?;
            tokio::fs::read(path).await?
        }
        "data" => {
            let data_url =
                data_url::DataUrl::process(url.as_str()).map_err(|e| anyhow::anyhow!("{:?}", e))?;
            let (data, _) = data_url
                .decode_to_vec()
                .map_err(|e| anyhow::anyhow!("{:?}", e))?;
            data
        }
        other => anyhow::bail!("Unsupported scheme: {}", other),
    };
    if let Ok(mut loading_progress) = loading_progress.lock() {
        loading_progress.loaded += 1;
    }
    Ok(data)
}

async fn generate_textures(
    url: &Url,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    loading_progress: &Arc<Mutex<LoadingProgress>>,
    doc: &gltf::Document,
    buffer_contents: &[Vec<u8>],
) -> Result<Vec<wgpu::Texture>, anyhow::Error> {
    let textures = futures::future::try_join_all(doc.images().map(|doc_image| {
        use gltf::image::Source;
        let source = doc_image.source();
        async {
            let contents = match source {
                Source::View { view, .. } => {
                    let parent_buffer_data = &buffer_contents[view.buffer().index()];
                    let begin = view.offset();
                    let end = begin + view.length();
                    let contents = &parent_buffer_data[begin..end];
                    Cow::Borrowed(contents)
                }
                Source::Uri { uri, .. } => {
                    let data = request_data(&url.join(uri)?, loading_progress.clone()).await?;
                    Cow::Owned(data)
                }
            };

            let dynamic_image = image::load_from_memory(&contents)?;

            // wgpu doesn't support 3 channel types, so I need to convert these
            let dynamic_image = match &dynamic_image {
                DynamicImage::ImageRgb8(_) => DynamicImage::ImageRgba8(dynamic_image.to_rgba8()),
                DynamicImage::ImageRgb16(_) => DynamicImage::ImageRgba16(dynamic_image.to_rgba16()),
                DynamicImage::ImageRgba32F(_) => {
                    DynamicImage::ImageRgba32F(dynamic_image.to_rgba32f())
                }
                _ => dynamic_image,
            };
            let format = match &dynamic_image {
                DynamicImage::ImageLuma8(_) => wgpu::TextureFormat::R8Unorm,
                DynamicImage::ImageLumaA8(_) => wgpu::TextureFormat::Rg8Unorm,
                DynamicImage::ImageRgba8(_) => wgpu::TextureFormat::Rgba8Unorm,
                DynamicImage::ImageLuma16(_) => wgpu::TextureFormat::R16Unorm,
                DynamicImage::ImageLumaA16(_) => wgpu::TextureFormat::Rg16Unorm,
                DynamicImage::ImageRgba16(_) => wgpu::TextureFormat::Rgba16Unorm,
                DynamicImage::ImageRgb32F(_) => wgpu::TextureFormat::Rgba32Float,
                other_format => {
                    return Err(anyhow::anyhow!("Unsupported format {:?}", other_format));
                }
            };
            let size: wgpu::Extent3d = wgpu::Extent3d {
                width: dynamic_image.width(),
                height: dynamic_image.height(),
                ..Default::default()
            };
            let mip_level_count = size.width.min(size.height).ilog2().max(1);
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size,
                mip_level_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[format.add_srgb_suffix(), format.remove_srgb_suffix()],
            });

            queue.write_texture(
                wgpu::TexelCopyTextureInfoBase {
                    texture: &texture,
                    mip_level: 0,
                    origin: Default::default(),
                    aspect: Default::default(),
                },
                dynamic_image.as_bytes(),
                wgpu::TexelCopyBufferLayout {
                    offset: Default::default(),
                    bytes_per_row: format.block_copy_size(None).map(|b| b * size.width),
                    rows_per_image: Default::default(),
                },
                size,
            );

            Ok::<_, anyhow::Error>(texture)
        }
    }))
    .await?;

    for texture in textures.iter() {
        let blitter = wgpu::util::TextureBlitterBuilder::new(device, texture.format())
            .sample_type(wgpu::FilterMode::Linear)
            .build();

        for base_mip_level in 1..texture.mip_level_count() {
            blitter.copy(
                device,
                encoder,
                &texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: base_mip_level - 1,
                    mip_level_count: Some(1),
                    ..Default::default()
                }),
                &texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level,
                    mip_level_count: Some(1),
                    ..Default::default()
                }),
            );
        }
    }

    Ok(textures)
}

// fn generate_materials(
//     device: &wgpu::Device,
//     doc: &gltf::Document,
//     textures: &[wgpu::Texture],
//     samplers: &[wgpu::Sampler],
//     default_texture: &wgpu::Texture,
//     default_sampler: &wgpu::Sampler,
// ) -> Vec<bind_groups::Material> {
//     doc.materials()
//         .map(|doc_material| {
//             let material_data = scene::Material {
//                 base_color_factor: glam::Vec4::from(
//                     doc_material.pbr_metallic_roughness().base_color_factor(),
//                 ),
//                 alpha_cutoff: doc_material.alpha_cutoff().unwrap_or(0.),
//                 _padding0: Default::default(),
//                 _padding1: Default::default(),
//                 _padding2: Default::default(),
//             };
//             let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//                 label: Some("Material Data"),
//                 contents: bytemuck::bytes_of(&material_data),
//                 usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
//             });

//             let doc_texture = doc_material
//                 .pbr_metallic_roughness()
//                 .base_color_texture()
//                 .map(|t| t.texture());
//             let base_color_texture = &doc_texture
//                 .as_ref()
//                 .map(|t| &textures[t.source().index()])
//                 .unwrap_or(default_texture)
//                 .create_view(&Default::default());
//             let base_color_sampler = doc_texture
//                 .and_then(|t| t.sampler().index())
//                 .map(|index| &samplers[index])
//                 .unwrap_or(default_sampler);
//             bind_groups::Material::from_bindings(
//                 device,
//                 bind_groups::MaterialLayout {
//                     material_data: material_buffer.as_entire_buffer_binding(),
//                     base_color_texture,
//                     base_color_sampler,
//                 },
//             )
//         })
//         .collect()
// }

fn generate_tlas(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    doc: &gltf::Document,
    mesh_blases: &[MeshBlases],
) -> wgpu::Tlas {
    // Get world transforms
    let mut nodes_to_visit = Vec::new();
    for doc_scene in doc.scenes() {
        nodes_to_visit.extend(doc_scene.nodes().map(|n| (n, Mat4::IDENTITY)));
    }
    let mut world_transforms = vec![Mat4::IDENTITY; doc.nodes().len()];
    while let Some((node, parent_transform)) = nodes_to_visit.pop() {
        let transform = Mat4::from_cols_array_2d(&node.transform().matrix());
        let world_transform = parent_transform * transform;
        world_transforms[node.index()] = world_transform;
        nodes_to_visit.extend(node.children().map(|n| (n, world_transform)));
    }
    let (bbox_min, bbox_max) = doc
        .nodes()
        .zip(world_transforms.iter())
        .filter_map(|(node, transform)| node.mesh().map(|m| (m, transform)))
        .flat_map(|(mesh, transform)| mesh.primitives().map(|p| (p.bounding_box(), *transform)))
        .fold((Vec3::MAX, Vec3::MIN), |(min, max), (bbox, transform)| {
            (
                min.min(transform.transform_point3(bbox.min.into())),
                max.max(transform.transform_point3(bbox.max.into())),
            )
        });

    let center = (bbox_min + bbox_max) * 0.5;
    let extent = bbox_max - bbox_min;
    let max_extent = extent.max_element();
    let scale_factor = max_extent.recip();

    let inv_bounding_box_matrix = Mat4::from_scale_rotation_translation(
        Vec3::splat(scale_factor),
        Quat::IDENTITY,
        scale_factor * -center,
    );

    for transform in world_transforms.iter_mut() {
        *transform = inv_bounding_box_matrix * *transform;
    }

    let mut tlas_instances: Vec<_> = Vec::new();
    doc.nodes()
        .zip(world_transforms.iter())
        .for_each(|(doc_node, &transform)| {
            doc_node.mesh().iter().for_each(|doc_mesh| {
                let primitive_blases = &mesh_blases[doc_mesh.index()].primitives;
                let transform = transform.transpose().to_cols_array()[..12]
                    .try_into()
                    .unwrap();
                tlas_instances.extend(doc_mesh.primitives().zip(primitive_blases.iter()).map({
                    |(doc_primitive, prim_blas)| {
                        wgpu::TlasInstance::new(
                            prim_blas,
                            transform,
                            doc_primitive.material().index().map_or(0, |i| i as u32 + 1),
                            0xFF,
                        )
                    }
                }));
            })
        });

    let doc_scene = doc.default_scene().or_else(|| doc.scenes().next()).unwrap();
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: doc_scene.name(),
        max_instances: tlas_instances.len() as _,
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
    });

    for (dst_instance, src_instance) in tlas
        .get_mut_slice(0..tlas_instances.len())
        .unwrap()
        .iter_mut()
        .zip(tlas_instances)
    {
        *dst_instance = Some(src_instance)
    }

    encoder.build_acceleration_structures(iter::empty(), iter::once(&tlas));

    tlas
}

struct MeshBlases {
    primitives: Vec<wgpu::Blas>,
}

fn generate_meshes(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    doc: &gltf::Document,
    buffers: &[wgpu::Buffer],
) -> Vec<MeshBlases> {
    use gltf::mesh::Semantic;

    doc.meshes()
        .map(|doc_mesh| {
            let primitives: Vec<_> = doc_mesh
                .primitives()
                .map(|doc_primitive| {
                    let (_, doc_positions) = doc_primitive
                        .attributes()
                        .find(|(s, _)| *s == Semantic::Positions)
                        .unwrap();

                    let view = doc_positions.view().unwrap();
                    let format = get_vertex_format(&doc_positions);
                    let vertex_stride = view.stride().map(|s| s as _).unwrap_or(format.size());
                    let vertex_buffer = &buffers[view.index()];
                    let first_vertex = (doc_positions.offset() as u64 / format.size()) as _;

                    let (index_format, index_count, index_buffer, first_index) =
                        if let Some(doc_indices) = doc_primitive.indices() {
                            use gltf::accessor::DataType;
                            let index_format = match doc_indices.data_type() {
                                DataType::U8 => wgpu::IndexFormat::Uint16,
                                DataType::U16 => wgpu::IndexFormat::Uint16,
                                DataType::U32 => wgpu::IndexFormat::Uint32,
                                t => unimplemented!("Index type {:?} is not supported", t),
                            };
                            let index_count = doc_indices.count() as u32;
                            let index_buffer = &buffers[doc_indices.view().unwrap().index()];
                            let first_index = doc_indices.offset() / index_format.byte_size();

                            (
                                Some(index_format),
                                Some(index_count),
                                Some(index_buffer),
                                Some(first_index as u32),
                            )
                        } else {
                            (None, None, None, None)
                        };

                    let size: wgpu::wgt::BlasTriangleGeometrySizeDescriptor =
                        wgpu::BlasTriangleGeometrySizeDescriptor {
                            vertex_format: format,
                            vertex_count: doc_positions.count() as _,
                            index_format,
                            index_count,
                            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
                        };

                    let blas = device.create_blas(
                        &wgpu::CreateBlasDescriptor {
                            label: doc_mesh.name(),
                            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD,
                            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                        },
                        wgpu::BlasGeometrySizeDescriptors::Triangles {
                            descriptors: vec![size.clone()],
                        },
                    );

                    let geometry = wgpu::BlasGeometries::TriangleGeometries(vec![
                        wgpu::BlasTriangleGeometry {
                            size: &size,
                            vertex_buffer,
                            first_vertex,
                            vertex_stride,
                            index_buffer,
                            first_index,
                            transform_buffer: None,
                            transform_buffer_offset: None,
                        },
                    ]);

                    encoder.build_acceleration_structures(
                        iter::once(&wgpu::BlasBuildEntry {
                            blas: &blas,
                            geometry,
                        }),
                        [],
                    );

                    blas
                })
                .collect();

            MeshBlases { primitives }
        })
        .collect()
}

fn get_vertex_format(accessor: &gltf::Accessor) -> wgpu::VertexFormat {
    use gltf::accessor::{DataType, Dimensions};
    match (
        accessor.normalized(),
        accessor.data_type(),
        accessor.dimensions(),
    ) {
        (true, DataType::I8, Dimensions::Vec2) => wgpu::VertexFormat::Snorm8x2,
        (true, DataType::I8, Dimensions::Vec4) => wgpu::VertexFormat::Snorm8x4,
        (true, DataType::U8, Dimensions::Vec2) => wgpu::VertexFormat::Unorm8x2,
        (true, DataType::U8, Dimensions::Vec4) => wgpu::VertexFormat::Unorm8x4,
        (true, DataType::I16, Dimensions::Vec2) => wgpu::VertexFormat::Snorm16x2,
        (true, DataType::I16, Dimensions::Vec4) => wgpu::VertexFormat::Snorm16x4,
        (true, DataType::U16, Dimensions::Vec2) => wgpu::VertexFormat::Unorm16x2,
        (true, DataType::U16, Dimensions::Vec4) => wgpu::VertexFormat::Unorm16x4,
        (false, DataType::I8, Dimensions::Vec2) => wgpu::VertexFormat::Sint8x2,
        (false, DataType::I8, Dimensions::Vec4) => wgpu::VertexFormat::Sint8x4,
        (false, DataType::U8, Dimensions::Vec2) => wgpu::VertexFormat::Uint8x2,
        (false, DataType::U8, Dimensions::Vec4) => wgpu::VertexFormat::Uint8x4,
        (false, DataType::I16, Dimensions::Vec2) => wgpu::VertexFormat::Sint16x2,
        (false, DataType::I16, Dimensions::Vec4) => wgpu::VertexFormat::Sint16x4,
        (false, DataType::U16, Dimensions::Vec2) => wgpu::VertexFormat::Uint16x2,
        (false, DataType::U16, Dimensions::Vec4) => wgpu::VertexFormat::Uint16x4,
        (false, DataType::U32, Dimensions::Scalar) => wgpu::VertexFormat::Uint32,
        (false, DataType::U32, Dimensions::Vec2) => wgpu::VertexFormat::Uint32x2,
        (false, DataType::U32, Dimensions::Vec3) => wgpu::VertexFormat::Uint32x3,
        (false, DataType::U32, Dimensions::Vec4) => wgpu::VertexFormat::Uint32x4,
        (_, DataType::F32, Dimensions::Scalar) => wgpu::VertexFormat::Float32,
        (_, DataType::F32, Dimensions::Vec2) => wgpu::VertexFormat::Float32x2,
        (_, DataType::F32, Dimensions::Vec3) => wgpu::VertexFormat::Float32x3,
        (_, DataType::F32, Dimensions::Vec4) => wgpu::VertexFormat::Float32x4,
        _ => unimplemented!(),
    }
}
