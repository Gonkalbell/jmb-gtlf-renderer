#![warn(clippy::all)]

mod app;
mod asset;
mod camera;
mod skybox;

mod raytracing;

#[allow(clippy::all)]
mod shaders;

use std::future::Future;

pub use app::RendererApp;

#[cfg(not(target_arch = "wasm32"))]
pub fn spawn(task: impl Future<Output = ()> + 'static + Send) {
    tokio::spawn(task);
}

#[cfg(target_arch = "wasm32")]
pub fn spawn(task: impl Future<Output = ()> + 'static) {
    wasm_bindgen_futures::spawn_local(task);
}

const ASSETS_BASE_URL: &str =
    "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/";

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub mod bind_groups {
    use super::shaders::*;

    pub type Camera = skybox::bind_groups::BindGroup0;
    pub type CameraLayout<'a> = skybox::bind_groups::BindGroupLayout0<'a>;

    pub type Material = scene::bind_groups::BindGroup1;
    pub type MaterialLayout<'a> = scene::bind_groups::BindGroupLayout1<'a>;

    pub type Instance = scene::bind_groups::BindGroup2;
    pub type InstanceLayout<'a> = scene::bind_groups::BindGroupLayout2<'a>;

    pub type Skybox = skybox::bind_groups::BindGroup1;
    pub type SkyboxLayout<'a> = skybox::bind_groups::BindGroupLayout1<'a>;

    pub type AccStructure = skybox::bind_groups::BindGroup2;
    pub type AccStructureLayout<'a> = skybox::bind_groups::BindGroupLayout2<'a>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedBufferSlice {
    buffer: wgpu::Buffer,
    offset: wgpu::BufferAddress,
    size: wgpu::BufferSize,
}

impl OwnedBufferSlice {
    fn from_slice(slice: &wgpu::BufferSlice) -> Self {
        Self {
            buffer: slice.buffer().clone(),
            offset: slice.offset(),
            size: slice.size(),
        }
    }

    fn as_slice<'a>(&'a self) -> wgpu::BufferSlice<'a> {
        self.buffer
            .slice(self.offset..self.offset + self.size.get())
    }
}
