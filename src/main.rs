#![warn(clippy::all, rust_2018_idioms)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use eframe::egui_wgpu::WgpuSetupCreateNew;

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
#[tokio::main]
async fn main() -> eframe::Result {
    use std::sync::Arc;

    use eframe::{NativeOptions, egui};

    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let native_options = NativeOptions {
        viewport: egui::ViewportBuilder::default().with_icon(
            // NOTE: Adding an icon is optional
            eframe::icon_data::from_png_bytes(&include_bytes!("../assets/icon-256.png")[..])
                .expect("Failed to load icon"),
        ),
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(
                eframe::egui_wgpu::WgpuSetupCreateNew {
                    device_descriptor: Arc::new(|_| wgpu::DeviceDescriptor {
                        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                        required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY
                            | wgpu::Features::EXPERIMENTAL_RAY_HIT_VERTEX_RETURN
                            | wgpu::Features::BUFFER_BINDING_ARRAY
                            | wgpu::Features::TEXTURE_BINDING_ARRAY
                            | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY,
                        required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                            .using_minimum_supported_acceleration_structure_values(),
                        ..Default::default()
                    }),
                    ..WgpuSetupCreateNew::without_display_handle()
                },
            ),

            ..Default::default()
        },
        depth_buffer: 32,
        ..Default::default()
    };
    eframe::run_native(
        "eframe template",
        native_options,
        Box::new(|cc| Ok(Box::new(jmb_gltf_renderer::RendererApp::new(cc)))),
    )
}

// When compiling to web using trunk:
#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::wasm_bindgen::JsCast as _;

    // Redirect `log` message to `console.log` and friends:
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    let web_options = eframe::WebOptions {
        wgpu_options: Default::default(),
        depth_buffer: 32,
        ..Default::default()
    };

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .expect("No window")
            .document()
            .expect("No document");

        let canvas = document
            .get_element_by_id("the_canvas_id")
            .expect("Failed to find the_canvas_id")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("the_canvas_id was not a HtmlCanvasElement");

        let start_result = eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|cc| Ok(Box::new(jmb_gltf_renderer::RendererApp::new(cc)))),
            )
            .await;

        // Remove the loading text and spinner:
        if let Some(loading_text) = document.get_element_by_id("loading_text") {
            match start_result {
                Ok(_) => {
                    loading_text.remove();
                }
                Err(e) => {
                    loading_text.set_inner_html(
                        "<p> The app has crashed. See the developer console for details. </p>",
                    );
                    panic!("Failed to start eframe: {e:?}");
                }
            }
        }
    });
}
