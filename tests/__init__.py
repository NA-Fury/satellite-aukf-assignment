import warnings

warnings.filterwarnings(
    "ignore",
    message="builtin type .* has no __module__ attribute",
    category=DeprecationWarning,
)
