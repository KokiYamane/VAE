import shutil


def print_progress_bar(i, length, width=None, end='\n', header=''):
    if width == None:
        terminal_size = shutil.get_terminal_size()
        width = terminal_size.columns-len(header)-len(str(i))-len(str(length))-7

    if i >= length - 1:
        progress_bar = '=' * width
        end = end
    else:
        num = round(i / (length-1) * width)
        progress_bar = '=' * (num-1) + '>' + ' ' * (width-num)
        end = ''
    print('\r\033[K{} [{}] {}/{}'.format(header, progress_bar, i+1, length), end=end)
