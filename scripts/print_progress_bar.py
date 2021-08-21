def print_progress_bar(i, length, width=30, end='\n', header=''):
    if i == length - 1:
        progress_bar = '=' * width
        end = end
    else:
        num = round(i / (length-1) * width)
        progress_bar = '=' * (num-1) + '>' + ' ' * (width-num)
        end = ''
    print('\r{} [{}] {}/{}'.format(header, progress_bar, i+1, length), end=end)
