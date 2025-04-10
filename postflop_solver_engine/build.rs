use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Get Python version from the virtual environment
    let venv_path = if let Ok(venv) = env::var("VIRTUAL_ENV") {
        PathBuf::from(venv)
    } else {
        // Fallback to looking for venv in parent directory
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
            .parent()
            .unwrap()
            .join("venv")
    };

    // Get Python version from the virtual environment's Python executable
    let python_version = Command::new(venv_path.join("bin/python"))
        .arg("-c")
        .arg("import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        .output()
        .expect("Failed to get Python version from virtual environment")
        .stdout;

    let python_version = String::from_utf8_lossy(&python_version).trim().to_string();
    
    // Try to find Python configuration
    let python_config = Command::new(format!("python{}-config", python_version))
        .arg("--ldflags")
        .output()
        .or_else(|_| {
            // Fallback to python3-config if specific version not found
            Command::new("python3-config")
                .arg("--ldflags")
                .output()
        })
        .expect("Failed to execute python-config. Make sure Python development files are installed.");
    
    let ldflags = String::from_utf8_lossy(&python_config.stdout);
    
    // Parse the library path from ldflags
    for flag in ldflags.split_whitespace() {
        if flag.starts_with("-L") {
            println!("cargo:rustc-link-search=native={}", &flag[2..]);
        }
    }
    
    // Add virtual environment library path
    println!("cargo:rustc-link-search=native={}", venv_path.join("lib").display());
    
    // Link against Python library
    println!("cargo:rustc-link-lib=python{}", python_version);
    println!("cargo:rustc-link-lib=dl");
    
    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=build.rs");
    
    // Tell cargo to rerun this if the environment changes
    println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");
} 