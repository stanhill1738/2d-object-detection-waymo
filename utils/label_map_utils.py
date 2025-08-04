def load_label_map(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("label_map", path)
    label_map_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(label_map_module)
    return label_map_module.label_map
