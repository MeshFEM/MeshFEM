def dependency_state(widget):
    import ipywidgets
    from ipywidgets import Widget
    # Patch the _find_widget_refs_by_state implementation to properly output,
    # e.g. the morphAttributes, which are stored as dictionary entries of
    # tuples of Widget
    def my_find_widget_refs_by_state(widget, state):
        """Find references to other widgets in a widget's state"""
        # Copy keys to allow changes to state during iteration:
        keys = tuple(state.keys())
        for key in keys:
            value = getattr(widget, key)
            # Trivial case: Direct references to other widgets:
            if isinstance(value, Widget):
                yield value
            # Also check for buried references in known, JSON-able structures
            # Note: This might miss references buried in more esoteric structures
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Widget):
                        yield item
            elif isinstance(value, dict):
                for item in value.values():
                    if isinstance(item, Widget):
                        yield item
                    # Also support tuples of Widgets within dicts
                    if isinstance(item, tuple):
                        for item2 in item:
                            if isinstance(item2, Widget):
                                yield item2
    ipywidgets.embed._find_widget_refs_by_state = my_find_widget_refs_by_state
    # https://github.com/jupyter-widgets/pythreejs/issues/217
    return ipywidgets.embed.dependency_state(widget)

def embed(path, widget):
    import ipywidgets
    ipywidgets.embed.embed_minimal_html(path, views=widget, state=dependency_state(widget))
