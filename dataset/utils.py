import re

def pre_caption(caption, max_words=30):
    """
    Preprocess caption text by cleaning and truncating to max_words.
    
    Args:
        caption: Input caption string
        max_words: Maximum number of words to keep
    
    Returns:
        Preprocessed caption string
    """
    caption = re.sub(
        r"([.!\"()*#:;~])",
        '',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')
    
    # Truncate to max_words
    words = caption.split(' ')
    if len(words) > max_words:
        caption = ' '.join(words[:max_words])
    
    return caption

