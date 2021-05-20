def print_banner(text: str, header_char: str = "-", footer_char: str = "-", size: int = 50) -> None:
    """
    Print a nicely formatted banner with a line of header and footer characters
    :param text:
    :param header_char:
    :param footer_char:
    :param size:
    :return:
    """
    print(header_char * size)
    print(text)
    print(footer_char * size)
