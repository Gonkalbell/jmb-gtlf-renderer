use wesl::{Mangler, Wesl};
use wgsl_to_wgpu::{
    MatrixVectorTypes, Module, ModulePath, TypePath, ValidationOptions, WgslCapabilities,
    WriteOptions,
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
        validate: Some(ValidationOptions {
            capabilities: WgslCapabilities::default(),
        }),
    };

    let mut module = Module::default();
    let root = ModulePath {
        components: vec!["skybox".to_owned()],
    };
    module.add_shader_module(
        &wesl.compile(&"package::skybox".parse()?)?.to_string(),
        None,
        options,
        root.clone(),
        |s| demangle_wesl(s, &root),
    )?;

    let root = ModulePath {
        components: vec!["scene".to_owned()],
    };
    module.add_shader_module(
        &wesl.compile(&"package::scene".parse()?)?.to_string(),
        None,
        options,
        root.clone(),
        |s| demangle_wesl(s, &root),
    )?;
    std::fs::write("src/shaders.rs", module.to_generated_bindings(options))?;

    Ok(())
}

pub fn demangle_wesl(name: &str, root: &ModulePath) -> TypePath {
    // Assume all paths are absolute paths.
    if name.starts_with("package_") {
        // Use the root module if unmangle fails.
        let mangler = wesl::EscapeMangler;

        let (path, name) = mangler.unmangle(name).unwrap();

        // Assume all wesl paths are absolute paths.
        TypePath {
            parent: ModulePath {
                components: path.components,
            },
            name,
        }
    } else {
        // Use the root module if the name is not mangled.
        wgsl_to_wgpu::TypePath {
            parent: root.clone(),
            name: name.to_string(),
        }
    }
}
