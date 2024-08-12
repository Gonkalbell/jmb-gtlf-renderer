mod camera;

#[allow(clippy::all)]
mod shaders;

use std::{collections::HashMap, f32::consts::TAU, primitive};

use eframe::wgpu;
use glam::{Mat4, Vec3A};
use puffin::profile_function;
use wgpu::util::DeviceExt;

use camera::ArcBallCamera;

use shaders::*;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

type CameraBindGroup = skybox::WgpuBindGroup0;
type CameraBindGroupEntries<'a> = skybox::WgpuBindGroup0Entries<'a>;
type CameraBindGroupEntriesParams<'a> = skybox::WgpuBindGroup0EntriesParams<'a>;

type NodeBindGroup = scene::WgpuBindGroup1;
type NodeBindGroupEntries<'a> = scene::WgpuBindGroup1Entries<'a>;
type NodeBindGroupEntriesParams<'a> = scene::WgpuBindGroup1EntriesParams<'a>;

type SkyboxBindGroup = skybox::WgpuBindGroup1;
type SkyboxBindGroupEntries<'a> = skybox::WgpuBindGroup1Entries<'a>;
type SkyboxBindGroupEntriesParams<'a> = skybox::WgpuBindGroup1EntriesParams<'a>;

struct Node {
    mesh_index: usize,
    bgroup: NodeBindGroup,
}

struct Primitive {
    pipeline: wgpu::RenderPipeline,
    buffers: Vec<wgpu::Buffer>,
    draw_count: u32,
    index_data: Option<PrimitiveIndexData>,
}

struct PrimitiveIndexData {
    buffer: wgpu::Buffer,
    // offset: u32,
    format: wgpu::IndexFormat,
}

pub struct SceneRenderer {
    camera_buf: wgpu::Buffer,
    user_camera: ArcBallCamera,

    // BIND GROUPS
    camera_bgroup: CameraBindGroup,
    skybox_bgroup: SkyboxBindGroup,

    // PIPELINES
    skybox_pipeline: wgpu::RenderPipeline,

    // dummy_primitive: Primitive,
    nodes: HashMap<usize, Node>,
    meshes: HashMap<usize, Vec<Primitive>>,
}

impl SceneRenderer {
    pub fn init(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        // Setup buffers and textures for the camera and skybox

        let user_camera = ArcBallCamera::default();

        let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::bytes_of(&bgroup_camera::Camera {
                view: Default::default(),
                view_inv: Default::default(),
                proj: Default::default(),
                proj_inv: Default::default(),
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let ktx_reader = ktx2::Reader::new(include_bytes!("../assets/rgba8.ktx2"))
            .expect("Failed to find skybox texture");
        let mut image = Vec::with_capacity(ktx_reader.data().len());
        for level in ktx_reader.levels() {
            image.extend_from_slice(level);
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

        // Create bind groups

        let camera_bgroup = CameraBindGroup::from_bindings(
            device,
            CameraBindGroupEntries::new(CameraBindGroupEntriesParams {
                res_camera: camera_buf.as_entire_buffer_binding(),
            }),
        );

        let skybox_bgroup = SkyboxBindGroup::from_bindings(
            device,
            SkyboxBindGroupEntries::new(SkyboxBindGroupEntriesParams {
                res_texture: &skybox_tview,
                res_sampler: &device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("skybox sampler"),
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                }),
            }),
        );

        // Create pipelines

        let shader = skybox::create_shader_module_embed_source(device);
        let skybox_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("skybox"),
            layout: Some(&skybox::create_pipeline_layout(device)),
            vertex: skybox::vertex_state(&shader, &skybox::vs_skybox_entry()),
            fragment: Some(skybox::fragment_state(
                &shader,
                &skybox::fs_skybox_entry([Some(color_format.into())]),
            )),
            primitive: wgpu::PrimitiveState {
                front_face: wgpu::FrontFace::Cw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Load the GLTF scene

        let (doc, buffers, images) =
            gltf::import("assets/models/AntiqueCamera/glTF/AntiqueCamera.gltf")
                .expect("Failed to load GLTF file");

        let shader = scene::create_shader_module_embed_source(device);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene primitive"),
            layout: Some(&scene::create_pipeline_layout(device)),
            vertex: scene::vertex_state(
                &shader,
                &scene::vs_scene_entry(wgpu::VertexStepMode::Vertex),
            ),
            fragment: Some(scene::fragment_state(
                &shader,
                &scene::fs_scene_entry([Some(color_format.into())]),
            )),
            primitive: wgpu::PrimitiveState {
                front_face: wgpu::FrontFace::Cw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let mut nodes = HashMap::new();
        let node_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DummyNode"),
            contents: bytemuck::bytes_of(&scene::Node {
                transform: Mat4::default(),
                normal_transform: Mat4::default(),
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bgroup = NodeBindGroup::from_bindings(
            device,
            NodeBindGroupEntries::new(NodeBindGroupEntriesParams {
                res_node: node_buf.as_entire_buffer_binding(),
            }),
        );
        nodes.insert(
            0,
            Node {
                bgroup,
                mesh_index: 0,
            },
        );

        let vertices = [
            scene::VertexInput::new(
                Vec3A::new(-1., -1., 0.),
                Vec3A::new(-1., -1., 1.).normalize(),
            ),
            scene::VertexInput::new(Vec3A::new(1., -1., 0.), Vec3A::new(1., -1., 1.).normalize()),
            scene::VertexInput::new(Vec3A::new(0., 1., 0.), Vec3A::new(0., 1., 1.).normalize()),
        ];
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DummyMesh"),
            contents: bytemuck::bytes_of(&vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let dummy_primitive = Primitive {
            pipeline,
            buffers: vec![vertex_buffer],
            draw_count: 3,
            index_data: None,
        };
        let mut meshes = HashMap::new();
        meshes.insert(0, vec![dummy_primitive]);

        Self {
            user_camera,
            camera_buf,

            camera_bgroup,
            skybox_bgroup,

            skybox_pipeline,

            nodes,
            meshes,
        }
    }

    pub fn prepare(
        &self,
        _device: &eframe::wgpu::Device,
        queue: &eframe::wgpu::Queue,
        _screen_descriptor: &eframe::egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut eframe::wgpu::CommandEncoder,
    ) -> Option<wgpu::CommandBuffer> {
        profile_function!();

        let view = self.user_camera.view_matrix();
        let proj = self.user_camera.projection_matrix();
        let camera = bgroup_camera::Camera {
            view,
            view_inv: view.inverse(),
            proj,
            proj_inv: proj.inverse(),
        };
        queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&camera));

        None
    }

    pub fn render<'a>(&'a self, rpass: &mut wgpu::RenderPass<'a>) {
        profile_function!();

        self.camera_bgroup.set(rpass);

        for node in self.nodes.values() {
            node.bgroup.set(rpass);
            let primitives = self
                .meshes
                .get(&node.mesh_index)
                .expect("Node didn't have a mesh");
            for primitive in primitives {
                rpass.set_pipeline(&primitive.pipeline);
                for (i, buffer) in primitive.buffers.iter().enumerate() {
                    rpass.set_vertex_buffer(i as _, buffer.slice(..));
                }
                if let Some(index_data) = &primitive.index_data {
                    rpass.set_index_buffer(index_data.buffer.slice(..), index_data.format);
                    rpass.draw_indexed(0..primitive.draw_count, 0, 0..1);
                } else {
                    rpass.draw(0..primitive.draw_count, 0..1);
                }
            }
        }

        self.skybox_bgroup.set(rpass);
        rpass.set_pipeline(&self.skybox_pipeline);
        rpass.draw(0..3, 0..1);
    }

    pub fn run_ui(&mut self, ctx: &egui::Context) {
        profile_function!();

        if !ctx.wants_keyboard_input() && !ctx.wants_pointer_input() {
            ctx.input(|input| {
                self.user_camera.update(input);
            });
        }

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    ui.menu_button("File", |ui| {
                        if ui.button("Quit").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });

                    // eframe doesn't support puffin on browser because it might not have a high resolution clock.
                    let mut are_scopes_on = puffin::are_scopes_on();
                    ui.toggle_value(&mut are_scopes_on, "Profiler");
                    puffin::set_scopes_on(are_scopes_on);
                }
                ui.menu_button("Camera", |ui| self.user_camera.run_ui(ui));
            });

            puffin_egui::show_viewport_if_enabled(ctx);
        });
    }
}
