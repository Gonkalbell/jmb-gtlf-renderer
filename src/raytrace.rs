use std::{iter, mem};

use glam::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

use crate::{
    DEPTH_FORMAT, bind_groups,
    shaders::{self, raytrace},
};

#[derive(Debug, Clone)]
pub struct Skybox {
    pub skybox_bgroup: bind_groups::Skybox,
    pub skybox_pipeline: wgpu::RenderPipeline,
    pub default_tlas: bind_groups::AccStructure,
}

impl Skybox {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
    ) -> Skybox {
        let default_tlas = default_acc_struct(device, queue);
        let ktx_reader = ktx2::Reader::new(include_bytes!("../assets/rgba8.ktx2"))
            .expect("Failed to find skybox texture");
        let mut image = Vec::with_capacity(ktx_reader.data().len());
        for level in ktx_reader.levels() {
            image.extend_from_slice(level.data);
        }
        let ktx_header = ktx_reader.header();
        let skybox_tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("../assets/rgba8.ktx2"),
                size: wgpu::Extent3d {
                    width: ktx_header.pixel_width,
                    height: ktx_header.pixel_height,
                    depth_or_array_layers: ktx_header.face_count,
                },
                mip_level_count: ktx_header.level_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::MipMajor,
            &image,
        );
        let skybox_tview = skybox_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("../assets/rgba8.ktx2"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..wgpu::TextureViewDescriptor::default()
        });

        let skybox_bgroup = bind_groups::Skybox::from_bindings(
            device,
            bind_groups::SkyboxLayout {
                res_texture: &skybox_tview,
                res_sampler: &device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("skybox sampler"),
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::MipmapFilterMode::Linear,
                    ..Default::default()
                }),
            },
        );

        let shader = raytrace::create_shader_module(device);
        let skybox_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("skybox"),
            layout: Some(&raytrace::create_pipeline_layout(device)),
            vertex: shaders::vertex_state(&shader, &raytrace::vs_skybox_entry()),
            fragment: Some(shaders::fragment_state(
                &shader,
                &raytrace::fs_skybox_entry([Some(color_format.into())]),
            )),
            primitive: wgpu::PrimitiveState {
                front_face: wgpu::FrontFace::Cw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            default_tlas,
            skybox_bgroup,
            skybox_pipeline,
        }
    }
}

fn create_vertices() -> (Vec<Vec4>, Vec<u16>) {
    let vertex_data = [
        // top (0, 0, 1)
        Vec4::new(-1., -1., 1., 1.),
        Vec4::new(1., -1., 1., 1.),
        Vec4::new(1., 1., 1., 1.),
        Vec4::new(-1., 1., 1., 1.),
        // bottom (0, 0, -1)
        Vec4::new(-1., 1., -1., 1.),
        Vec4::new(1., 1., -1., 1.),
        Vec4::new(1., -1., -1., 1.),
        Vec4::new(-1., -1., -1., 1.),
        // right (1., 0, 0)
        Vec4::new(1., -1., -1., 1.),
        Vec4::new(1., 1., -1., 1.),
        Vec4::new(1., 1., 1., 1.),
        Vec4::new(1., -1., 1., 1.),
        // left (-1., 0, 0)
        Vec4::new(-1., -1., 1., 1.),
        Vec4::new(-1., 1., 1., 1.),
        Vec4::new(-1., 1., -1., 1.),
        Vec4::new(-1., -1., -1., 1.),
        // front (0, 1., 0)
        Vec4::new(1., 1., -1., 1.),
        Vec4::new(-1., 1., -1., 1.),
        Vec4::new(-1., 1., 1., 1.),
        Vec4::new(1., 1., 1., 1.),
        // back (0, -1., 0)
        Vec4::new(1., -1., 1., 1.),
        Vec4::new(-1., -1., 1., 1.),
        Vec4::new(-1., -1., -1., 1.),
        Vec4::new(1., -1., -1., 1.),
    ];

    let index_data: &[u16] = &[
        0, 1, 2, 2, 3, 0, // top
        4, 5, 6, 6, 7, 4, // bottom
        8, 9, 10, 10, 11, 8, // right
        12, 13, 14, 14, 15, 12, // left
        16, 17, 18, 18, 19, 16, // front
        20, 21, 22, 22, 23, 20, // back
    ];

    (vertex_data.to_vec(), index_data.to_vec())
}

pub fn default_acc_struct(device: &wgpu::Device, queue: &wgpu::Queue) -> bind_groups::AccStructure {
    let (vertex_data, index_data) = create_vertices();

    let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertex_data),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::BLAS_INPUT,
    });

    let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Index Buffer"),
        contents: bytemuck::cast_slice(&index_data),
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::BLAS_INPUT,
    });

    let blas_geo_size_desc = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: vertex_data.len() as u32,
        index_format: Some(wgpu::IndexFormat::Uint16),
        index_count: Some(index_data.len() as u32),
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };

    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: None,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_geo_size_desc.clone()],
        },
    );

    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: None,
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        max_instances: 1,
    });

    let bind_group_layout = bind_groups::AccStructureLayout { acc_struct: &tlas };

    let bind_group = bind_groups::AccStructure::from_bindings(device, bind_group_layout);

    let instance = &mut tlas[0];

    let transform = Mat4::from_scale(Vec3::splat(0.5))
        .transpose()
        .to_cols_array()[..12]
        .try_into()
        .unwrap();

    *instance = Some(wgpu::TlasInstance::new(&blas, transform, 0, 0xff));

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    encoder.build_acceleration_structures(
        iter::once(&wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &blas_geo_size_desc,
                vertex_buffer: &vertex_buf,
                first_vertex: 0,
                vertex_stride: mem::size_of::<Vec4>() as u64,
                index_buffer: Some(&index_buf),
                first_index: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }),
        iter::once(&tlas),
    );

    queue.submit(Some(encoder.finish()));

    bind_group
}
