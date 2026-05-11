use anyhow::Result;
use wesl::{Mangler, Wesl};
use wgsl_to_wgpu::{
    MatrixVectorTypes, Module, TypePath, WriteOptions,
};

// src/build.rs
fn main() -> anyhow::Result<()> {
    let wesl = Wesl::new("src/shaders");

    let options = WriteOptions {
        derive_bytemuck_vertex: true,
        derive_bytemuck_host_shareable: true,
        derive_encase_host_shareable: false,
        derive_serde: true,
        matrix_vector_types: MatrixVectorTypes::Glam,
        rustfmt: true,
        validate: None,
    };

    let mut bindings_module = Module::default();
    add_entry_point_module(&mut bindings_module, &wesl, options, "package::skybox".parse()?)?;
    add_entry_point_module(&mut bindings_module, &wesl, options, "package::scene".parse()?)?;
    // add_entry_point_module(&mut bindings_module, &wesl, options, "package::raytracing".parse()?)?;

    std::fs::write("src/shaders.rs", bindings_module.to_generated_bindings(options))?;

    Ok(())
}

fn add_entry_point_module(
    bindings_module: &mut Module,
    wesl: &Wesl<wesl::StandardResolver>,
    options: WriteOptions,
    root: wesl::ModulePath,
) -> Result<()> {
    let compiled = &wesl.compile(&root)?;
    wesl::emit_rerun_if_changed(&compiled.modules, wesl.resolver());

    bindings_module.add_shader_module(
        &compiled.to_string(),
        None,
        options,
        convert_wesl_module_path(root.clone()),
        |s| demangle_wesl(s, &root),
    )?;
    Ok(())
}

pub fn convert_wesl_module_path(path: wesl::ModulePath) -> wgsl_to_wgpu::ModulePath {
    wgsl_to_wgpu::ModulePath {
        components: path.components,
    }
}

pub fn demangle_wesl(name: &str, root: &wesl::ModulePath) -> TypePath {
    // Assume all paths are absolute paths.
    if name.starts_with("package_") {
        // Use the root module if unmangle fails.
        let mangler = wesl::EscapeMangler;

        let (path, name) = mangler.unmangle(name).unwrap();

        // Assume all wesl paths are absolute paths.
        TypePath {
            parent: convert_wesl_module_path(path),
            name,
        }
    } else {
        // Use the root module if the name is not mangled.
        wgsl_to_wgpu::TypePath {
            parent: convert_wesl_module_path(root.clone()),
            name: name.to_string(),
        }
    }
}
