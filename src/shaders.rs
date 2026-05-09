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
pub mod scene {
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
                    visibility: wgpu::ShaderStages::VERTEX,
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
            pub material_data: wgpu::BufferBinding<'a>,
            pub base_color_texture: &'a wgpu::TextureView,
            pub base_color_sampler: &'a wgpu::Sampler,
        }
        const LAYOUT_DESCRIPTOR1: wgpu::BindGroupLayoutDescriptor =
            wgpu::BindGroupLayoutDescriptor {
                label: Some("LayoutDescriptor1"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
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
                            resource: wgpu::BindingResource::Buffer(bindings.material_data),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(
                                bindings.base_color_texture,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(bindings.base_color_sampler),
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
            pub res_instances: wgpu::BufferBinding<'a>,
        }
        const LAYOUT_DESCRIPTOR2: wgpu::BindGroupLayoutDescriptor =
            wgpu::BindGroupLayoutDescriptor {
                label: Some("LayoutDescriptor2"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
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
                        resource: wgpu::BindingResource::Buffer(bindings.res_instances),
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
    pub fn fs_scene_entry(targets: [Option<wgpu::ColorTargetState>; 1]) -> super::FragmentEntry<1> {
        super::FragmentEntry {
            entry_point: ENTRY_FS_SCENE,
            targets,
            constants: Default::default(),
        }
    }
    pub const SOURCE : & str = "struct Material {\n    base_color_factor: vec4f,\n    alpha_cutoff: f32\n}\n\n@group(1) @binding(0)\nvar<uniform> material_data: Material;\n\n@group(1) @binding(2)\nvar base_color_texture: texture_2d<f32>;\n\n@group(1) @binding(1)\nvar base_color_sampler: sampler;\n\nstruct Instance {\n    local_to_world: mat4x4f,\n    normal_local_to_world: mat4x4f\n}\n\n@group(2) @binding(0)\nvar<storage> res_instances: array<Instance>;\n\nstruct VertexInput {\n    @location(0)\n    position: vec3f,\n    @location(1)\n    normal: vec3f,\n    @location(2)\n    tangent: vec4f,\n    @location(3)\n    texcoord_0: vec2f,\n    @location(4)\n    texcoord_1: vec2f,\n    @location(5)\n    color_0: vec4f,\n    @location(6)\n    color_1: vec4f\n}\n\nstruct VertexOutput {\n    @builtin(position)\n    position: vec4f,\n    @location(0)\n    view_position: vec4f,\n    @location(1)\n    normal: vec3f,\n    @location(2)\n    tangent: vec3f,\n    @location(3)\n    texcoord_0: vec2f,\n    @location(4)\n    texcoord_1: vec2f,\n    @location(5)\n    color_0: vec4f,\n    @location(6)\n    color_1: vec4f\n}\n\n@vertex\nfn vs_scene(@builtin(instance_index) instance_index: u32, input: VertexInput) -> VertexOutput {\n    var output: VertexOutput;\n    var instance = res_instances[instance_index];\n    output.position = package__1bgroup_camera__1res_camera.world_to_proj * instance.local_to_world * vec4f(input.position, 1);\n    output.view_position = package__1bgroup_camera__1res_camera.world_to_local * instance.local_to_world * vec4f(input.position, 1);\n    output.normal = (package__1bgroup_camera__1res_camera.world_to_local * instance.normal_local_to_world * vec4f(input.normal, 0)).xyz;\n    output.tangent = (package__1bgroup_camera__1res_camera.world_to_local * instance.normal_local_to_world * vec4f(input.tangent.xyz, 0)).xyz;\n    output.texcoord_0 = input.texcoord_0;\n    output.texcoord_1 = input.texcoord_1;\n    output.color_0 = input.color_0;\n    output.color_1 = input.color_1;\n    return output;\n}\n\nconst LIGHT_DIR = vec3f(0.25, 0.5, 1);\n\nconst AMBIENT_COLOR = vec3f(0.1);\n\n@fragment\nfn fs_scene(input: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4f {\n    let base_color = input.color_0 * textureSample(base_color_texture, base_color_sampler, input.texcoord_0) * material_data.base_color_factor;\n    if (base_color.a < material_data.alpha_cutoff) {\n        discard;\n    }\n    var N = input.normal;\n    if all(N == vec3f(0)) {\n        let view_position = input.view_position.xyz;\n        let dx = dpdx(view_position);\n        let dy = dpdy(view_position);\n        N = cross(dy, dx);\n    }\n    N = select(-1.0, 1.0, front_facing) * normalize(N);\n    let L = normalize(LIGHT_DIR);\n    let NDotL = max(dot(N, L), 0.0);\n    let surface_color = (base_color.rgb * AMBIENT_COLOR) + (base_color.rgb * NDotL);\n    return vec4f(surface_color, base_color.a);\n}\n\nstruct package__1bgroup_camera_Camera {\n    world_to_local: mat4x4<f32>,\n    local_to_world: mat4x4<f32>,\n    local_to_proj: mat4x4<f32>,\n    proj_to_local: mat4x4<f32>,\n    world_to_proj: mat4x4<f32>,\n    proj_to_world: mat4x4<f32>\n}\n\n@group(0) @binding(0)\nvar<uniform> package__1bgroup_camera__1res_camera: package__1bgroup_camera_Camera;\n" ;
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
    pub const ENTRY_FS_SCENE: &str = "fs_scene";
    pub const ENTRY_VS_SCENE: &str = "vs_scene";
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
    pub struct Instance {
        pub local_to_world: glam::Mat4,
        pub normal_local_to_world: glam::Mat4,
    }
    const _: () = assert!(
        std::mem::size_of::<Instance>() == 128,
        "size of Instance does not match WGSL"
    );
    const _: () = assert!(
        std::mem::offset_of!(Instance, local_to_world) == 0,
        "offset of Instance.local_to_world does not match WGSL"
    );
    const _: () = assert!(
        std::mem::offset_of!(Instance, normal_local_to_world) == 64,
        "offset of Instance.normal_local_to_world does not match WGSL"
    );
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
    pub struct Material {
        pub base_color_factor: glam::Vec4,
        pub alpha_cutoff: f32,
    }
    const _: () = assert!(
        std::mem::size_of::<Material>() == 32,
        "size of Material does not match WGSL"
    );
    const _: () = assert!(
        std::mem::offset_of!(Material, base_color_factor) == 0,
        "offset of Material.base_color_factor does not match WGSL"
    );
    const _: () = assert!(
        std::mem::offset_of!(Material, alpha_cutoff) == 16,
        "offset of Material.alpha_cutoff does not match WGSL"
    );
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
    pub struct VertexInput {
        pub position: glam::Vec3,
        pub normal: glam::Vec3,
        pub tangent: glam::Vec4,
        pub texcoord_0: glam::Vec2,
        pub texcoord_1: glam::Vec2,
        pub color_0: glam::Vec4,
        pub color_1: glam::Vec4,
    }
    impl VertexInput {
        pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 7] = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: std::mem::offset_of!(VertexInput, position) as u64,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: std::mem::offset_of!(VertexInput, normal) as u64,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: std::mem::offset_of!(VertexInput, tangent) as u64,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: std::mem::offset_of!(VertexInput, texcoord_0) as u64,
                shader_location: 3,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: std::mem::offset_of!(VertexInput, texcoord_1) as u64,
                shader_location: 4,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: std::mem::offset_of!(VertexInput, color_0) as u64,
                shader_location: 5,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: std::mem::offset_of!(VertexInput, color_1) as u64,
                shader_location: 6,
            },
        ];
        pub const fn vertex_buffer_layout(
            step_mode: wgpu::VertexStepMode,
        ) -> wgpu::VertexBufferLayout<'static> {
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<VertexInput>() as u64,
                step_mode,
                attributes: &VertexInput::VERTEX_ATTRIBUTES,
            }
        }
    }
    pub fn vs_scene_entry(input_step_mode: wgpu::VertexStepMode) -> super::VertexEntry<1> {
        super::VertexEntry {
            entry_point: ENTRY_VS_SCENE,
            buffers: [VertexInput::vertex_buffer_layout(input_step_mode)],
            constants: Default::default(),
        }
    }
}
pub mod skybox {
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
                    visibility: wgpu::ShaderStages::VERTEX,
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
        #[derive(Debug, Copy, Clone)]
        pub struct BindGroups<'a> {
            pub bind_group0: &'a BindGroup0,
            pub bind_group1: &'a BindGroup1,
        }
        impl BindGroups<'_> {
            pub fn set<P: super::super::SetBindGroup>(&self, pass: &mut P) {
                self.bind_group0.set(pass);
                self.bind_group1.set(pass);
            }
        }
    }
    pub fn set_bind_groups<P: super::SetBindGroup>(
        pass: &mut P,
        bind_group0: &bind_groups::BindGroup0,
        bind_group1: &bind_groups::BindGroup1,
    ) {
        bind_group0.set(pass);
        bind_group1.set(pass);
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
    pub const SOURCE : & str = "@group(1) @binding(0)\nvar res_texture: texture_cube<f32>;\n\n@group(1) @binding(1)\nvar res_sampler: sampler;\n\nstruct SkyboxInterp {\n    @builtin(position)\n    position: vec4<f32>,\n    @location(0)\n    tex_coord: vec3<f32>\n}\n\n@vertex\nfn vs_skybox(@builtin(vertex_index) vertex_index: u32) -> SkyboxInterp {\n    let tmp1 = i32(vertex_index) / 2;\n    let tmp2 = i32(vertex_index) & 1;\n    let pos = vec4<f32>(f32(tmp1) * 4.0 - 1.0, f32(tmp2) * 4.0 - 1.0, 1.0, 1.0);\n    var result: SkyboxInterp;\n    result.position = pos;\n    let dir = vec4<f32>((package__1bgroup_camera__1res_camera.proj_to_local * pos).xyz, 0.0);\n    result.tex_coord = (package__1bgroup_camera__1res_camera.local_to_world * dir).xyz;\n    return result;\n}\n\n@fragment\nfn fs_skybox(vertex: SkyboxInterp) -> @location(0) vec4<f32> {\n    return textureSample(res_texture, res_sampler, vertex.tex_coord);\n}\n\nstruct package__1bgroup_camera_Camera {\n    world_to_local: mat4x4<f32>,\n    local_to_world: mat4x4<f32>,\n    local_to_proj: mat4x4<f32>,\n    proj_to_local: mat4x4<f32>,\n    world_to_proj: mat4x4<f32>,\n    proj_to_world: mat4x4<f32>\n}\n\n@group(0) @binding(0)\nvar<uniform> package__1bgroup_camera__1res_camera: package__1bgroup_camera_Camera;\n" ;
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
