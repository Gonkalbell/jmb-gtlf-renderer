
#![allow(warnings)]
#![allow(clippy::all)]
#[derive(Debug)]
pub struct VertexEntry<const N: usize> {
    pub entry_point: &'static str,
    pub buffers: [wgpu::VertexBufferLayout<'static>; N],
    pub constants: Vec<(&'static str, f64)>,
}
pub fn vertex_state<'a, const N: usize>(
    module: &'a wgpu::ShaderModule,
    entry: &'a VertexEntry<N>,
) -> wgpu::VertexState<'a> {
    wgpu::VertexState {
        module,
        entry_point: Some(entry.entry_point),
        buffers: &entry.buffers,
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &entry.constants,
            ..Default::default()
        },
    }
}
#[derive(Debug)]
pub struct FragmentEntry<const N: usize> {
    pub entry_point: &'static str,
    pub targets: [Option<wgpu::ColorTargetState>; N],
    pub constants: Vec<(&'static str, f64)>,
}
pub fn fragment_state<'a, const N: usize>(
    module: &'a wgpu::ShaderModule,
    entry: &'a FragmentEntry<N>,
) -> wgpu::FragmentState<'a> {
    wgpu::FragmentState {
        module,
        entry_point: Some(entry.entry_point),
        targets: &entry.targets,
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &entry.constants,
            ..Default::default()
        },
    }
}
pub trait SetBindGroup {
    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &wgpu::BindGroup,
        offsets: &[wgpu::DynamicOffset],
    );
}
impl SetBindGroup for wgpu::ComputePass<'_> {
    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &wgpu::BindGroup,
        offsets: &[wgpu::DynamicOffset],
    ) {
        self.set_bind_group(index, bind_group, offsets);
    }
}
impl SetBindGroup for wgpu::RenderPass<'_> {
    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &wgpu::BindGroup,
        offsets: &[wgpu::DynamicOffset],
    ) {
        self.set_bind_group(index, bind_group, offsets);
    }
}
impl SetBindGroup for wgpu::RenderBundleEncoder<'_> {
    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &wgpu::BindGroup,
        offsets: &[wgpu::DynamicOffset],
    ) {
        self.set_bind_group(index, bind_group, offsets);
    }
}
pub mod bgroup_camera {
    #[repr(C)]
    #[derive(
        Debug,
        Copy,
        Clone,
        PartialEq,
        bytemuck :: Pod,
        bytemuck :: Zeroable,
        serde :: Serialize,
        serde :: Deserialize,
    )]
    pub struct Camera {
        pub world_to_local: glam::Mat4,
        pub local_to_world: glam::Mat4,
        pub local_to_proj: glam::Mat4,
        pub proj_to_local: glam::Mat4,
        pub world_to_proj: glam::Mat4,
        pub proj_to_world: glam::Mat4,
    }
    const _: () = assert!(
        std::mem::size_of::<Camera>() == 384,
        "size of Camera does not match WGSL"
    );
    const _: () = assert!(
        std::mem::offset_of!(Camera, world_to_local) == 0,
        "offset of Camera.world_to_local does not match WGSL"
    );
    const _: () = assert!(
        std::mem::offset_of!(Camera, local_to_world) == 64,
        "offset of Camera.local_to_world does not match WGSL"
    );
    const _: () = assert!(
        std::mem::offset_of!(Camera, local_to_proj) == 128,
        "offset of Camera.local_to_proj does not match WGSL"
    );
    const _: () = assert!(
        std::mem::offset_of!(Camera, proj_to_local) == 192,
        "offset of Camera.proj_to_local does not match WGSL"
    );
    const _: () = assert!(
        std::mem::offset_of!(Camera, world_to_proj) == 256,
        "offset of Camera.world_to_proj does not match WGSL"
    );
    const _: () = assert!(
        std::mem::offset_of!(Camera, proj_to_world) == 320,
        "offset of Camera.proj_to_world does not match WGSL"
    );
}
pub mod raytrace {
    pub mod bind_groups {
        #[derive(Debug, Clone)]
        pub struct BindGroup0(wgpu::BindGroup);
        #[derive(Debug)]
        pub struct BindGroupLayout0<'a> {
            pub res_camera: wgpu::BufferBinding<'a>,
        }
        const LAYOUT_DESCRIPTOR0: wgpu::BindGroupLayoutDescriptor =
            wgpu::BindGroupLayoutDescriptor {
                label: Some("LayoutDescriptor0"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            };
        impl BindGroup0 {
            pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
                device.create_bind_group_layout(&LAYOUT_DESCRIPTOR0)
            }
            pub fn from_bindings(device: &wgpu::Device, bindings: BindGroupLayout0) -> Self {
                let bind_group_layout = device.create_bind_group_layout(&LAYOUT_DESCRIPTOR0);
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(bindings.res_camera),
                    }],
                    label: Some("BindGroup0"),
                });
                Self(bind_group)
            }
            pub fn set<P: super::super::SetBindGroup>(&self, pass: &mut P) {
                pass.set_bind_group(0, &self.0, &[]);
            }
            pub fn inner(&self) -> &wgpu::BindGroup {
                &self.0
            }
        }
        #[derive(Debug, Clone)]
        pub struct BindGroup1(wgpu::BindGroup);
        #[derive(Debug)]
        pub struct BindGroupLayout1<'a> {
            pub res_texture: &'a wgpu::TextureView,
            pub res_sampler: &'a wgpu::Sampler,
        }
        const LAYOUT_DESCRIPTOR1: wgpu::BindGroupLayoutDescriptor =
            wgpu::BindGroupLayoutDescriptor {
                label: Some("LayoutDescriptor1"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            };
        impl BindGroup1 {
            pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
                device.create_bind_group_layout(&LAYOUT_DESCRIPTOR1)
            }
            pub fn from_bindings(device: &wgpu::Device, bindings: BindGroupLayout1) -> Self {
                let bind_group_layout = device.create_bind_group_layout(&LAYOUT_DESCRIPTOR1);
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(bindings.res_texture),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(bindings.res_sampler),
                        },
                    ],
                    label: Some("BindGroup1"),
                });
                Self(bind_group)
            }
            pub fn set<P: super::super::SetBindGroup>(&self, pass: &mut P) {
                pass.set_bind_group(1, &self.0, &[]);
            }
            pub fn inner(&self) -> &wgpu::BindGroup {
                &self.0
            }
        }
        #[derive(Debug, Clone)]
        pub struct BindGroup2(wgpu::BindGroup);
        #[derive(Debug)]
        pub struct BindGroupLayout2<'a> {
            pub acc_struct: &'a wgpu::Tlas,
        }
        const LAYOUT_DESCRIPTOR2: wgpu::BindGroupLayoutDescriptor =
            wgpu::BindGroupLayoutDescriptor {
                label: Some("LayoutDescriptor2"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::AccelerationStructure {
                        vertex_return: false,
                    },
                    count: None,
                }],
            };
        impl BindGroup2 {
            pub fn get_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
                device.create_bind_group_layout(&LAYOUT_DESCRIPTOR2)
            }
            pub fn from_bindings(device: &wgpu::Device, bindings: BindGroupLayout2) -> Self {
                let bind_group_layout = device.create_bind_group_layout(&LAYOUT_DESCRIPTOR2);
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::AccelerationStructure(bindings.acc_struct),
                    }],
                    label: Some("BindGroup2"),
                });
                Self(bind_group)
            }
            pub fn set<P: super::super::SetBindGroup>(&self, pass: &mut P) {
                pass.set_bind_group(2, &self.0, &[]);
            }
            pub fn inner(&self) -> &wgpu::BindGroup {
                &self.0
            }
        }
        #[derive(Debug, Copy, Clone)]
        pub struct BindGroups<'a> {
            pub bind_group0: &'a BindGroup0,
            pub bind_group1: &'a BindGroup1,
            pub bind_group2: &'a BindGroup2,
        }
        impl BindGroups<'_> {
            pub fn set<P: super::super::SetBindGroup>(&self, pass: &mut P) {
                self.bind_group0.set(pass);
                self.bind_group1.set(pass);
                self.bind_group2.set(pass);
            }
        }
    }
    pub fn set_bind_groups<P: super::SetBindGroup>(
        pass: &mut P,
        bind_group0: &bind_groups::BindGroup0,
        bind_group1: &bind_groups::BindGroup1,
        bind_group2: &bind_groups::BindGroup2,
    ) {
        bind_group0.set(pass);
        bind_group1.set(pass);
        bind_group2.set(pass);
    }
    pub fn fs_skybox_entry(
        targets: [Option<wgpu::ColorTargetState>; 1],
    ) -> super::FragmentEntry<1> {
        super::FragmentEntry {
            entry_point: ENTRY_FS_SKYBOX,
            targets,
            constants: Default::default(),
        }
    }
    pub const SOURCE : & str = "enable wgpu_ray_query;\n\n@group(1) @binding(0)\nvar res_texture: texture_cube<f32>;\n\n@group(1) @binding(1)\nvar res_sampler: sampler;\n\n@group(2) @binding(0)\nvar acc_struct: acceleration_structure;\n\nstruct SkyboxInterp {\n    @builtin(position)\n    position: vec4f,\n    @location(0)\n    ray_dir: vec3<f32>\n}\n\n@vertex\nfn vs_skybox(@builtin(vertex_index) vertex_index: u32) -> SkyboxInterp {\n    let x = i32(vertex_index) / 2;\n    let y = i32(vertex_index) & 1;\n    let tc = vec2<f32>(f32(x) * 2.0, f32(y) * 2.0);\n    let pos = vec4f(1.0 - 2.0 * tc, 0.0, 1.0);\n    var result: SkyboxInterp;\n    result.position = pos;\n    let dir = vec4f((package__1bgroup_camera__1res_camera.proj_to_local * pos).xyz, 0.0);\n    result.ray_dir = (package__1bgroup_camera__1res_camera.local_to_world * dir).xyz;\n    return result;\n}\n\nfn u32x3_to_color(p: vec3<u32>) -> vec3f {\n    let r = reverseBits(p.x);\n    let g = reverseBits(p.y);\n    let b = reverseBits(p.z);\n    return vec3f(f32(r), f32(g), f32(b)) * (1.0 / f32(4294967295u));\n}\n\n@fragment\nfn fs_skybox(vertex: SkyboxInterp) -> @location(0) vec4f {\n    let origin = (package__1bgroup_camera__1res_camera.local_to_world * vec4f(0.0, 0.0, 0.0, 1.0)).xyz;\n    var rq: ray_query;\n    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 255u, 0.01, 200.0, origin, vertex.ray_dir));\n    rayQueryProceed(&rq);\n    let intersection = rayQueryGetCommittedIntersection(&rq);\n    var result = vec4f(0.0);\n    if intersection.kind == RAY_QUERY_INTERSECTION_NONE {\n        result = textureSample(res_texture, res_sampler, vertex.ray_dir);\n    }\n    else {\n        let w = 1.0 - intersection.barycentrics.x - intersection.barycentrics.y;\n        let centrality = min(min(intersection.barycentrics.x, intersection.barycentrics.y), w) * 3.0;\n        let input = vec3<u32>(intersection.geometry_index, intersection.primitive_index, intersection.instance_custom_data);\n        let color = u32x3_to_color(input);\n        result = select(0.0, 1.0, centrality > 0.1) * vec4f(color, 1.0);\n    }\n    return result;\n}\n\nstruct package__1bgroup_camera_Camera {\n    world_to_local: mat4x4<f32>,\n    local_to_world: mat4x4<f32>,\n    local_to_proj: mat4x4<f32>,\n    proj_to_local: mat4x4<f32>,\n    world_to_proj: mat4x4<f32>,\n    proj_to_world: mat4x4<f32>\n}\n\n@group(0) @binding(0)\nvar<uniform> package__1bgroup_camera__1res_camera: package__1bgroup_camera_Camera;\n" ;
    pub fn create_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
        let source = std::borrow::Cow::Borrowed(SOURCE);
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(source),
        })
    }
    pub fn create_pipeline_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[
                Some(&bind_groups::BindGroup0::get_bind_group_layout(device)),
                Some(&bind_groups::BindGroup1::get_bind_group_layout(device)),
                Some(&bind_groups::BindGroup2::get_bind_group_layout(device)),
            ],
            immediate_size: 0,
        })
    }
    pub const ENTRY_FS_SKYBOX: &str = "fs_skybox";
    pub const ENTRY_VS_SKYBOX: &str = "vs_skybox";
    pub fn vs_skybox_entry() -> super::VertexEntry<0> {
        super::VertexEntry {
            entry_point: ENTRY_VS_SKYBOX,
            buffers: [],
            constants: Default::default(),
        }
    }
}

    