
def title(text:str) -> str:
    """Convert a string to title case, replacing underscores with spaces.
    
    Args:
        text: Input string.
        
    Returns:
        Title-cased string.
    """
    txt = "-"*40 + "\n"
    txt += text + "\n"
    txt += "-"*40
    return txt

def tabbed(text:str, n_tabs:int=1) -> str:
    """Indent each line of the input text by a specified number of tabs.
    
    Args:
        text: Input string.
        n_tabs: Number of tabs to indent.
        
    Returns:
        Indented string.
    """
    tab_str = "\t" * n_tabs
    return tab_str + text
