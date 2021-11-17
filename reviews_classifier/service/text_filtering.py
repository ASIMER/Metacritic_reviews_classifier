
def preprocess(text):
    """preprocess template"""
    # lowercase
    text = text.lower()
    # remove digits
    text = text.replace(r"\b\d+\b", ' ')
    # remove whitespaces
    text = ' '.join(text.replace("\xa0", " ").split())
    # remove html tags
    pattern = r"""(?x)                              # Turn on free-spacing
      <[^>]+>                                       # Remove <html> tags
      | &([a-z0-9]+|\#[0-9]{1,6}|\#x[0-9a-f]{1,6}); # Remove &nbsp;
      """
    text = text.replace(pattern, "")
    # replace urls
    text = text.replace(r"http\S+", " ")
    return text
