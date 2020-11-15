'''
To process CORRUPT EXIF data warning, run this code.
If you encounter an error, do the following.
1. apt-get update
2. apt-get install imagemagick
3. Convert png file to jpg. Refer to
https://discuss.pytorch.org/t/corrupt-exif-data-messages-when-training-imagenet/17313
'''
import glob
import piexif


nfiles=0
for filename in glob.iglob('/home/osilab/dataset/ILSVRC2015/ILSVRC2015/Data/CLS-LOC/train/**/*.JPEG', recursive=True):
    nfiles += 1
    print("About to process file %d, which is %s." %(nfiles, filename))
    piexif.remove(filename)
