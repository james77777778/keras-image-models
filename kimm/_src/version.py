from kimm._src.kimm_export import kimm_export

__version__ = "0.2.2"


@kimm_export("kimm")
def version():
    return __version__
