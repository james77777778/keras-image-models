try:
    import namex
except ImportError:
    namex = None

# These dicts reference "canonical names" only
# (i.e. the first name an object was registered with).
REGISTERED_NAMES_TO_OBJS = {}
REGISTERED_OBJS_TO_NAMES = {}


def register_internal_serializable(path, symbol):
    global REGISTERED_NAMES_TO_OBJS
    if isinstance(path, (list, tuple)):
        name = path[0]
    else:
        name = path
    REGISTERED_NAMES_TO_OBJS[name] = symbol
    REGISTERED_OBJS_TO_NAMES[symbol] = name


def get_symbol_from_name(name):
    return REGISTERED_NAMES_TO_OBJS.get(name, None)


def get_name_from_symbol(symbol):
    return REGISTERED_OBJS_TO_NAMES.get(symbol, None)


if namex:

    class kimm_export:
        def __init__(self, parent_path):
            package = "kimm"

            if isinstance(parent_path, str):
                export_paths = [parent_path]
            elif isinstance(parent_path, list):
                export_paths = parent_path
            else:
                raise ValueError(
                    f"Invalid type for `parent_path` argument: "
                    f"Received '{parent_path}' "
                    f"of type {type(parent_path)}"
                )
            for p in export_paths:
                if not p.startswith(package):
                    raise ValueError(
                        "All `export_path` values should start with "
                        f"'{package}.'. Received: parent_path={parent_path}"
                    )
            self.package = package
            self.parent_path = parent_path

        def __call__(self, symbol):
            if hasattr(symbol, "_api_export_path") and (
                symbol._api_export_symbol_id == id(symbol)
            ):
                raise ValueError(
                    f"Symbol {symbol} is already exported as "
                    f"'{symbol._api_export_path}'. "
                    f"Cannot also export it to '{self.parent_path}'."
                )
            if isinstance(self.parent_path, list):
                path = [p + f".{symbol.__name__}" for p in self.parent_path]
            elif isinstance(self.parent_path, str):
                path = self.parent_path + f".{symbol.__name__}"
            symbol._api_export_path = path
            symbol._api_export_symbol_id = id(symbol)

            register_internal_serializable(path, symbol)
            return symbol

else:

    class kimm_export:
        def __init__(self, parent_path):
            self.parent_path = parent_path

        def __call__(self, symbol):
            if isinstance(self.parent_path, list):
                path = [p + f".{symbol.__name__}" for p in self.parent_path]
            elif isinstance(self.parent_path, str):
                path = self.parent_path + f".{symbol.__name__}"

            register_internal_serializable(path, symbol)
            return symbol
