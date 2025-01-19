# cvProj

usage: cereali.py [options] <product images directory> <shelf image>

positional arguments:
prod_img_dir Path to the input image directory
shelf_img Path to the test image

options:
-h, --help show this help message and exit
-s, --save_output Save output images
-t {a,b,all}, --test {a,b,all}
test to apply
-m, --multi detect multiple products

# example to use

python cereali.py -t all models/ scenes/e1.png
