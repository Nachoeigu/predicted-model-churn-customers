def replace_yes_no_with_boolean(data):
    if (data == 'Yes')|(data =='Male'):
        return 1
    elif (data == 'No')|(data =='Female'):
        return 0
