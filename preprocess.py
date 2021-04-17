import os
root = 'data'
folder = os.listdir(root)
txt = open('data.txt', 'a')
for f in folder:
    if '.jpg' in f:
        f_align = f.split('.')[0]
        aligns = os.listdir('{}/{}'.format(root, f_align))
        for align in aligns:
            txt.write('{}/{} {}/{}/{}\n'.format(root, f, root, f_align, align))
txt.close()
