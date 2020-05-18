#include "data.h"
#include "utils.h"
#include "image.h"
#include "cuda.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

extern int check_mistakes;

list *get_paths(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    list *lines = make_list();
    while((path=fgetl(file))){
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}

/*
char **get_random_paths_indexes(char **paths, int n, int m, int *indexes)
{
    char **random_paths = calloc(n, sizeof(char*));
    int i;
    pthread_mutex_lock(&mutex);
    for(i = 0; i < n; ++i){
        int index = rand()%m;
        indexes[i] = index;
        random_paths[i] = paths[index];
        if(i == 0) printf("%s\n", paths[index]);
    }
    pthread_mutex_unlock(&mutex);
    return random_paths;
}
*/

char **get_random_paths(char **paths, int n, int m)
{
    char **random_paths = calloc(n, sizeof(char*));
    int i;
    pthread_mutex_lock(&mutex);
    for(i = 0; i < n; ++i){
        int index = rand()%m;
        random_paths[i] = paths[index];
        //if(i == 0) printf("%s\n", paths[index]);
    }
    pthread_mutex_unlock(&mutex);
    return random_paths;
}

char **find_replace_paths(char **paths, int n, char *find, char *replace)
{
    char **replace_paths = calloc(n, sizeof(char*));
    int i;
    for(i = 0; i < n; ++i){
        char replaced[4096];
        find_replace(paths[i], find, replace, replaced);
        replace_paths[i] = copy_string(replaced);
    }
    return replace_paths;
}

matrix load_image_paths_gray(char **paths, int n, int w, int h)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image(paths[i], w, h, 3);

        image gray = grayscale_image(im);
        free_image(im);
        im = gray;

        X.vals[i] = im.data;
        X.cols = im.h*im.w*im.c;
    }
    return X;
}

matrix load_image_paths(char **paths, int n, int w, int h)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], w, h);
        X.vals[i] = im.data;
        X.cols = im.h*im.w*im.c;
    }
    return X;
}

matrix load_image_augment_paths(char **paths, int n, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], 0, 0);
        image crop;
        if(center){
            crop = center_crop_image(im, size, size);
        } else {
            crop = random_augment_image(im, angle, aspect, min, max, size, size);
        }
        int flip = rand()%2;
        if (flip) flip_image(crop);
        random_distort_image(crop, hue, saturation, exposure);

        /*
        show_image(im, "orig");
        show_image(crop, "crop");
        cvWaitKey(0);
        */
        //grayscale_image_3c(crop);
        free_image(im);
        X.vals[i] = crop.data;
        X.cols = crop.h*crop.w*crop.c;
    }
    return X;
}


box_label *read_boxes(char *filename, int *n)
{
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    float x, y, h, w;
    int id;
    int count = 0;
    int size = 64;
    box_label *boxes = calloc(size, sizeof(box_label));
    while(fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5){
        if(count == size) {
            size = size * 2;
            boxes = realloc(boxes, size*sizeof(box_label));
        }
        boxes[count].id = id;
        boxes[count].x = x;
        boxes[count].y = y;
        boxes[count].h = h;
        boxes[count].w = w;
        boxes[count].left   = x - w/2;
        boxes[count].right  = x + w/2;
        boxes[count].top    = y - h/2;
        boxes[count].bottom = y + h/2;
        ++count;
    }
    fclose(file);
    *n = count;
    return boxes;
}

void randomize_boxes(box_label *b, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        box_label swap = b[i];
        int index = rand()%n;
        b[i] = b[index];
        b[index] = swap;
    }
}

void correct_boxes(box_label *boxes, int n, float dx, float dy, float sx, float sy, int flip)
{
    int i;
    for(i = 0; i < n; ++i){
        if(boxes[i].x == 0 && boxes[i].y == 0) {
            boxes[i].x = 999999;
            boxes[i].y = 999999;
            boxes[i].w = 999999;
            boxes[i].h = 999999;
            continue;
        }
        boxes[i].left   = boxes[i].left  * sx - dx;
        boxes[i].right  = boxes[i].right * sx - dx;
        boxes[i].top    = boxes[i].top   * sy - dy;
        boxes[i].bottom = boxes[i].bottom* sy - dy;

        if(flip){
            float swap = boxes[i].left;
            boxes[i].left = 1. - boxes[i].right;
            boxes[i].right = 1. - swap;
        }

        boxes[i].left =  constrain(0, 1, boxes[i].left);
        boxes[i].right = constrain(0, 1, boxes[i].right);
        boxes[i].top =   constrain(0, 1, boxes[i].top);
        boxes[i].bottom =   constrain(0, 1, boxes[i].bottom);

        boxes[i].x = (boxes[i].left+boxes[i].right)/2;
        boxes[i].y = (boxes[i].top+boxes[i].bottom)/2;
        boxes[i].w = (boxes[i].right - boxes[i].left);
        boxes[i].h = (boxes[i].bottom - boxes[i].top);

        boxes[i].w = constrain(0, 1, boxes[i].w);
        boxes[i].h = constrain(0, 1, boxes[i].h);
    }
}

void fill_truth_swag(char *path, float *truth, int classes, int flip, float dx, float dy, float sx, float sy)
{
    char labelpath[4096];
    find_replace(path, "images", "labels", labelpath);
    find_replace(labelpath, "JPEGImages", "labels", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);

    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    float x,y,w,h;
    int id;
    int i;

    for (i = 0; i < count && i < 90; ++i) {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if (w < .0 || h < .0) continue;

        int index = (4+classes) * i;

        truth[index++] = x;
        truth[index++] = y;
        truth[index++] = w;
        truth[index++] = h;

        if (id < classes) truth[index+id] = 1;
    }
    free(boxes);
}

void fill_truth_region(char *path, float *truth, int classes, int num_boxes, int flip, float dx, float dy, float sx, float sy)
{
    char labelpath[4096];
    find_replace(path, "images", "labels", labelpath);
    find_replace(labelpath, "JPEGImages", "labels", labelpath);

    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".png", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    float x,y,w,h;
    int id;
    int i;

    for (i = 0; i < count; ++i) {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if (w < .005 || h < .005) continue;

        int col = (int)(x*num_boxes);
        int row = (int)(y*num_boxes);

        x = x*num_boxes - col;
        y = y*num_boxes - row;

        int index = (col+row*num_boxes)*(5+classes);
        if (truth[index]) continue;
        truth[index++] = 1;

        if (id < classes) truth[index+id] = 1;
        index += classes;

        truth[index++] = x;
        truth[index++] = y;
        truth[index++] = w;
        truth[index++] = h;
    }
    free(boxes);
}

void load_rle(image im, int *rle, int n)
{
    int count = 0;
    int curr = 0;
    int i,j;
    for(i = 0; i < n; ++i){
        for(j = 0; j < rle[i]; ++j){
            im.data[count++] = curr;
        }
        curr = 1 - curr;
    }
    for(; count < im.h*im.w*im.c; ++count){
        im.data[count] = curr;
    }
}

void or_image(image src, image dest, int c)
{
    int i;
    for(i = 0; i < src.w*src.h; ++i){
        if(src.data[i]) dest.data[dest.w*dest.h*c + i] = 1;
    }
}

void exclusive_image(image src)
{
    int k, j, i;
    int s = src.w*src.h;
    for(k = 0; k < src.c-1; ++k){
        for(i = 0; i < s; ++i){
            if (src.data[k*s + i]){
                for(j = k+1; j < src.c; ++j){
                    src.data[j*s + i] = 0;
                }
            }
        }
    }
}

box bound_image(image im)
{
    int x,y;
    int minx = im.w;
    int miny = im.h;
    int maxx = 0;
    int maxy = 0;
    for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            if(im.data[y*im.w + x]){
                minx = (x < minx) ? x : minx;
                miny = (y < miny) ? y : miny;
                maxx = (x > maxx) ? x : maxx;
                maxy = (y > maxy) ? y : maxy;
            }
        }
    }
    box b = {minx, miny, maxx-minx + 1, maxy-miny + 1};
    //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
    return b;
}

void fill_truth_iseg(char *path, int num_boxes, float *truth, int classes, int w, int h, augment_args aug, int flip, int mw, int mh)
{
    char labelpath[4096];
    find_replace(path, "images", "mask", labelpath);
    find_replace(labelpath, "JPEGImages", "mask", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    FILE *file = fopen(labelpath, "r");
    if(!file) file_error(labelpath);
    char buff[32788];
    int id;
    int i = 0;
    int j;
    image part = make_image(w, h, 1);
    while((fscanf(file, "%d %s", &id, buff) == 2) && i < num_boxes){
        int n = 0;
        int *rle = read_intlist(buff, &n, 0);
        load_rle(part, rle, n);
        image sized = rotate_crop_image(part, aug.rad, aug.scale, aug.w, aug.h, aug.dx, aug.dy, aug.aspect);
        if(flip) flip_image(sized);

        image mask = resize_image(sized, mw, mh);
        truth[i*(mw*mh+1)] = id;
        for(j = 0; j < mw*mh; ++j){
            truth[i*(mw*mh + 1) + 1 + j] = mask.data[j];
        }
        ++i;

        free_image(mask);
        free_image(sized);
        free(rle);
    }
    if(i < num_boxes) truth[i*(mw*mh+1)] = -1;
    fclose(file);
    free_image(part);
}

void fill_truth_mask(char *path, int num_boxes, float *truth, int classes, int w, int h, augment_args aug, int flip, int mw, int mh)
{
    char labelpath[4096];
    find_replace(path, "images", "mask", labelpath);
    find_replace(labelpath, "JPEGImages", "mask", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    FILE *file = fopen(labelpath, "r");
    if(!file) file_error(labelpath);
    char buff[32788];
    int id;
    int i = 0;
    image part = make_image(w, h, 1);
    while((fscanf(file, "%d %s", &id, buff) == 2) && i < num_boxes){
        int n = 0;
        int *rle = read_intlist(buff, &n, 0);
        load_rle(part, rle, n);
        image sized = rotate_crop_image(part, aug.rad, aug.scale, aug.w, aug.h, aug.dx, aug.dy, aug.aspect);
        if(flip) flip_image(sized);
        box b = bound_image(sized);
        if(b.w > 0){
            image crop = crop_image(sized, b.x, b.y, b.w, b.h);
            image mask = resize_image(crop, mw, mh);
            truth[i*(4 + mw*mh + 1) + 0] = (b.x + b.w/2.)/sized.w;
            truth[i*(4 + mw*mh + 1) + 1] = (b.y + b.h/2.)/sized.h;
            truth[i*(4 + mw*mh + 1) + 2] = b.w/sized.w;
            truth[i*(4 + mw*mh + 1) + 3] = b.h/sized.h;
            int j;
            for(j = 0; j < mw*mh; ++j){
                truth[i*(4 + mw*mh + 1) + 4 + j] = mask.data[j];
            }
            truth[i*(4 + mw*mh + 1) + 4 + mw*mh] = id;
            free_image(crop);
            free_image(mask);
            ++i;
        }
        free_image(sized);
        free(rle);
    }
    fclose(file);
    free_image(part);
}

void trim(char *str)
{
    char* buffer = (char*)calloc(8192, sizeof(char));
    sprintf(buffer, "%s", str);

    char *p = buffer;
    while (*p == ' ' || *p == '\t') ++p;

    char *end = p + strlen(p) - 1;
    while (*end == ' ' || *end == '\t') {
        *end = '\0';
        --end;
    }
    sprintf(str, "%s", p);

    free(buffer);
}

void find_replace_extension(char *str, char *orig, char *rep, char *output)
{
    char* buffer = (char*)calloc(8192, sizeof(char));

    sprintf(buffer, "%s", str);
    char *p = strstr(buffer, orig);
    int offset = (p - buffer);
    int chars_from_end = strlen(buffer) - offset;
    if (!p || chars_from_end != strlen(orig)) {  // Is 'orig' even in 'str' AND is 'orig' found at the end of 'str'?
        sprintf(output, "%s", buffer);
        free(buffer);
        return;
    }

    *p = '\0';
    sprintf(output, "%s%s%s", buffer, rep, p + strlen(orig));
    free(buffer);
}


void replace_image_to_label(char* input_path, char* output_path)
{
    find_replace(input_path, "/images/train2014/", "/labels/train2014/", output_path);    // COCO
    find_replace(output_path, "/images/val2014/", "/labels/val2014/", output_path);        // COCO
    find_replace(output_path, "/JPEGImages/", "/labels/", output_path);    // PascalVOC
    find_replace(output_path, "\\images\\train2014\\", "\\labels\\train2014\\", output_path);    // COCO
    find_replace(output_path, "\\images\\val2014\\", "\\labels\\val2014\\", output_path);        // COCO
    find_replace(output_path, "\\JPEGImages\\", "\\labels\\", output_path);    // PascalVOC
    //find_replace(output_path, "/images/", "/labels/", output_path);    // COCO
    //find_replace(output_path, "/VOC2007/JPEGImages/", "/VOC2007/labels/", output_path);        // PascalVOC
    //find_replace(output_path, "/VOC2012/JPEGImages/", "/VOC2012/labels/", output_path);        // PascalVOC

    //find_replace(output_path, "/raw/", "/labels/", output_path);
    trim(output_path);

    // replace only ext of files
    find_replace_extension(output_path, ".jpg", ".txt", output_path);
    find_replace_extension(output_path, ".JPG", ".txt", output_path); // error
    find_replace_extension(output_path, ".jpeg", ".txt", output_path);
    find_replace_extension(output_path, ".JPEG", ".txt", output_path);
    find_replace_extension(output_path, ".png", ".txt", output_path);
    find_replace_extension(output_path, ".PNG", ".txt", output_path);
    find_replace_extension(output_path, ".bmp", ".txt", output_path);
    find_replace_extension(output_path, ".BMP", ".txt", output_path);
    find_replace_extension(output_path, ".ppm", ".txt", output_path);
    find_replace_extension(output_path, ".PPM", ".txt", output_path);
    find_replace_extension(output_path, ".tiff", ".txt", output_path);
    find_replace_extension(output_path, ".TIFF", ".txt", output_path);

    // Check file ends with txt:
    if(strlen(output_path) > 4) {
        char *output_path_ext = output_path + strlen(output_path) - 4;
        if( strcmp(".txt", output_path_ext) != 0){
            fprintf(stderr, "Failed to infer label file name (check image extension is supported): %s \n", output_path);
        }
    }else{
        fprintf(stderr, "Label file name is too short: %s \n", output_path);
    }
}

int fill_truth_detection(char *path, int num_boxes, float *truth, int classes, int flip, float dx, float dy, float sx, float sy,
    int net_w, int net_h)
{
    char labelpath[4096];
    replace_image_to_label(path, labelpath);

    int count = 0;
    int i;
    box_label *boxes = read_boxes(labelpath, &count);
    int min_w_h = 0;
    float lowest_w = 1.F / net_w;
    float lowest_h = 1.F / net_h;
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    if (count > num_boxes) count = num_boxes;
    float x, y, w, h;
    int id;
    int sub = 0;

    for (i = 0; i < count; ++i) {
        x = boxes[i].x;
        y = boxes[i].y;
        w = boxes[i].w;
        h = boxes[i].h;
        id = boxes[i].id;

        // not detect small objects
        //if ((w < 0.001F || h < 0.001F)) continue;
        // if truth (box for object) is smaller than 1x1 pix
        char buff[256];
        if (id >= classes) {
            printf("\n Wrong annotation: class_id = %d. But class_id should be [from 0 to %d], file: %s \n", id, (classes-1), labelpath);
            sprintf(buff, "echo %s \"Wrong annotation: class_id = %d. But class_id should be [from 0 to %d]\" >> bad_label.list", labelpath, id, (classes-1));
            system(buff);
            if (check_mistakes) getchar();
            ++sub;
            continue;
        }
        if ((w < lowest_w || h < lowest_h)) {
            //sprintf(buff, "echo %s \"Very small object: w < lowest_w OR h < lowest_h\" >> bad_label.list", labelpath);
            //system(buff);
            ++sub;
            continue;
        }
        if (x == 999999 || y == 999999) {
            printf("\n Wrong annotation: x = 0, y = 0, < 0 or > 1, file: %s \n", labelpath);
            sprintf(buff, "echo %s \"Wrong annotation: x = 0 or y = 0\" >> bad_label.list", labelpath);
            system(buff);
            ++sub;
            if (check_mistakes) getchar();
            continue;
        }
        if (x <= 0 || x > 1 || y <= 0 || y > 1) {
            printf("\n Wrong annotation: x = %f, y = %f, file: %s \n", x, y, labelpath);
            sprintf(buff, "echo %s \"Wrong annotation: x = %f, y = %f\" >> bad_label.list", labelpath, x, y);
            system(buff);
            ++sub;
            if (check_mistakes) getchar();
            continue;
        }
        if (w > 1) {
            printf("\n Wrong annotation: w = %f, file: %s \n", w, labelpath);
            sprintf(buff, "echo %s \"Wrong annotation: w = %f\" >> bad_label.list", labelpath, w);
            system(buff);
            w = 1;
            if (check_mistakes) getchar();
        }
        if (h > 1) {
            printf("\n Wrong annotation: h = %f, file: %s \n", h, labelpath);
            sprintf(buff, "echo %s \"Wrong annotation: h = %f\" >> bad_label.list", labelpath, h);
            system(buff);
            h = 1;
            if (check_mistakes) getchar();
        }
        if (x == 0) x += lowest_w;
        if (y == 0) y += lowest_h;

        truth[(i-sub)*5+0] = x;
        truth[(i-sub)*5+1] = y;
        truth[(i-sub)*5+2] = w;
        truth[(i-sub)*5+3] = h;
        truth[(i-sub)*5+4] = id;

        if (min_w_h == 0) min_w_h = w*net_w;
        if (min_w_h > w*net_w) min_w_h = w*net_w;
        if (min_w_h > h*net_h) min_w_h = h*net_h;
    }
    free(boxes);
    return min_w_h;
}


/*void fill_truth_detection(char *path, int num_boxes, float *truth, int classes, int flip, float dx, float dy, float sx, float sy)
{
    char labelpath[4096];
    find_replace(path, "images", "labels", labelpath);
    find_replace(labelpath, "JPEGImages", "labels", labelpath);
    find_replace(labelpath, "image_2", "labels", labelpath);  // KITTI

    find_replace(labelpath, "raw", "labels", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".png", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    if(count > num_boxes) count = num_boxes;
    float x,y,w,h;
    int id;
    int i;
    int sub = 0;

    for (i = 0; i < count; ++i) {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if ((w < .001 || h < .001)) {
            ++sub;
            continue;
        }

        truth[(i-sub)*5+0] = x;
        truth[(i-sub)*5+1] = y;
        truth[(i-sub)*5+2] = w;
        truth[(i-sub)*5+3] = h;
        truth[(i-sub)*5+4] = id;
    }
    free(boxes);
}*/

#define NUMCHARS 37

void print_letters(float *pred, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        int index = max_index(pred+i*NUMCHARS, NUMCHARS);
        printf("%c", int_to_alphanum(index));
    }
    printf("\n");
}

void fill_truth_captcha(char *path, int n, float *truth)
{
    char *begin = strrchr(path, '/');
    ++begin;
    int i;
    for(i = 0; i < strlen(begin) && i < n && begin[i] != '.'; ++i){
        int index = alphanum_to_int(begin[i]);
        if(index > 35) printf("Bad %c\n", begin[i]);
        truth[i*NUMCHARS+index] = 1;
    }
    for(;i < n; ++i){
        truth[i*NUMCHARS + NUMCHARS-1] = 1;
    }
}

data load_data_captcha(char **paths, int n, int m, int k, int w, int h)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = make_matrix(n, k*NUMCHARS);
    int i;
    for(i = 0; i < n; ++i){
        fill_truth_captcha(paths[i], k, d.y.vals[i]);
    }
    if(m) free(paths);
    return d;
}

data load_data_captcha_encode(char **paths, int n, int m, int w, int h)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.X.cols = 17100;
    d.y = d.X;
    if(m) free(paths);
    return d;
}

void fill_truth(char *path, char **labels, int k, float *truth)
{
    int i;
    memset(truth, 0, k*sizeof(float));
    int count = 0;
    char *multi[1024];
    char long_labels[1024];
    for(i = 0; i < k; ++i){
        sprintf(long_labels,"/%s/",labels[i]);
        //if(strstr(path, labels[i])){
        if(strstr(path, long_labels)){
            multi[count] = labels[i];
            truth[i] = 1;
            ++count;
            //printf("%s %s %d\n", path, labels[i], i);
        }
    }
    if(count != 1 && (k != 1 || count != 0)){
        printf("Too many or too few labels: %d, %s\n", count, path);
        for(i=0;i<count;i++) printf("count %d k %d label %s\n",i,k,multi[i]);
    }
}

void fill_hierarchy(float *truth, int k, tree *hierarchy)
{
    int j;
    for(j = 0; j < k; ++j){
        if(truth[j]){
            int parent = hierarchy->parent[j];
            while(parent >= 0){
                truth[parent] = 1;
                parent = hierarchy->parent[parent];
            }
        }
    }
    int i;
    int count = 0;
    for(j = 0; j < hierarchy->groups; ++j){
        //printf("%d\n", count);
        int mask = 1;
        for(i = 0; i < hierarchy->group_size[j]; ++i){
            if(truth[count + i]){
                mask = 0;
                break;
            }
        }
        if (mask) {
            for(i = 0; i < hierarchy->group_size[j]; ++i){
                truth[count + i] = SECRET_NUM;
            }
        }
        count += hierarchy->group_size[j];
    }
}

matrix load_regression_labels_paths(char **paths, int n, int k)
{
    matrix y = make_matrix(n, k);
    int i,j;
    for(i = 0; i < n; ++i){
        char labelpath[4096];
        find_replace(paths[i], "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".BMP", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);
        find_replace(labelpath, ".JPG", ".txt", labelpath);
        find_replace(labelpath, ".JPeG", ".txt", labelpath);
        find_replace(labelpath, ".Jpeg", ".txt", labelpath);
        find_replace(labelpath, ".PNG", ".txt", labelpath);
        find_replace(labelpath, ".TIF", ".txt", labelpath);
        find_replace(labelpath, ".bmp", ".txt", labelpath);
        find_replace(labelpath, ".jpeg", ".txt", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".png", ".txt", labelpath);
        find_replace(labelpath, ".tif", ".txt", labelpath);

        FILE *file = fopen(labelpath, "r");
        for(j = 0; j < k; ++j){
            fscanf(file, "%f", &(y.vals[i][j]));
        }
        fclose(file);
    }
    return y;
}

matrix load_labels_paths(char **paths, int n, char **labels, int k, tree *hierarchy)
{
    matrix y = make_matrix(n, k);
    int i;
    for(i = 0; i < n && labels; ++i){
        fill_truth(paths[i], labels, k, y.vals[i]);
        if(hierarchy){
            fill_hierarchy(y.vals[i], k, hierarchy);
        }
    }
    return y;
}

matrix load_tags_paths(char **paths, int n, int k)
{
    matrix y = make_matrix(n, k);
    int i;
    //int count = 0;
    for(i = 0; i < n; ++i){
        char label[4096];
        find_replace(paths[i], "images", "labels", label);
        find_replace(label, ".jpg", ".txt", label);
        FILE *file = fopen(label, "r");
        if (!file) continue;
        //++count;
        int tag;
        while(fscanf(file, "%d", &tag) == 1){
            if(tag < k){
                y.vals[i][tag] = 1;
            }
        }
        fclose(file);
    }
    //printf("%d/%d\n", count, n);
    return y;
}

char **get_labels(char *filename)
{
    list *plist = get_paths(filename);
    char **labels = (char **)list_to_array(plist);
    free_list(plist);
    return labels;
}

void free_data(data d)
{
    if(!d.shallow){
        free_matrix(d.X);
        free_matrix(d.y);
    }else{
        free(d.X.vals);
        free(d.y.vals);
    }
}

image get_segmentation_image(char *path, int w, int h, int classes)
{
    char labelpath[4096];
    find_replace(path, "images", "mask", labelpath);
    find_replace(labelpath, "JPEGImages", "mask", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    image mask = make_image(w, h, classes);
    FILE *file = fopen(labelpath, "r");
    if(!file) file_error(labelpath);
    char buff[32788];
    int id;
    image part = make_image(w, h, 1);
    while(fscanf(file, "%d %s", &id, buff) == 2){
        int n = 0;
        int *rle = read_intlist(buff, &n, 0);
        load_rle(part, rle, n);
        or_image(part, mask, id);
        free(rle);
    }
    //exclusive_image(mask);
    fclose(file);
    free_image(part);
    return mask;
}

image get_segmentation_image2(char *path, int w, int h, int classes)
{
    char labelpath[4096];
    find_replace(path, "images", "mask", labelpath);
    find_replace(labelpath, "JPEGImages", "mask", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    image mask = make_image(w, h, classes+1);
    int i;
    for(i = 0; i < w*h; ++i){
        mask.data[w*h*classes + i] = 1;
    }
    FILE *file = fopen(labelpath, "r");
    if(!file) file_error(labelpath);
    char buff[32788];
    int id;
    image part = make_image(w, h, 1);
    while(fscanf(file, "%d %s", &id, buff) == 2){
        int n = 0;
        int *rle = read_intlist(buff, &n, 0);
        load_rle(part, rle, n);
        or_image(part, mask, id);
        for(i = 0; i < w*h; ++i){
            if(part.data[i]) mask.data[w*h*classes + i] = 0;
        }
        free(rle);
    }
    //exclusive_image(mask);
    fclose(file);
    free_image(part);
    return mask;
}

data load_data_seg(int n, char **paths, int m, int w, int h, int classes, int min, int max, float angle, float aspect, float hue, float saturation, float exposure, int div)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;


    d.y.rows = n;
    d.y.cols = h*w*classes/div/div;
    d.y.vals = calloc(d.X.rows, sizeof(float*));

    for(i = 0; i < n; ++i){
        char labelpath[4096];
        find_replace(random_paths[i], "SegmentationClass", "JPEGImages", labelpath);
        find_replace(labelpath, ".png", ".jpg", labelpath);
        image orig = load_image_color(labelpath, 0, 0);
        //image orig = load_image_color(random_paths[i], 0, 0);
        augment_args a = random_augment_args(orig, angle, aspect, min, max, w, h);
        image sized = rotate_crop_image(orig, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);

        int flip = rand()%2;
        if(flip) flip_image(sized);
        random_distort_image(sized, hue, saturation, exposure);
        d.X.vals[i] = sized.data;

//        image mask = get_segmentation_image(random_paths[i], orig.w, orig.h, classes);
        image mask = load_image(random_paths[i], orig.w, orig.h, 1);
        //image mask = make_image(orig.w, orig.h, classes+1);
        image sized_m = rotate_crop_image(mask, a.rad, a.scale/div, a.w/div, a.h/div, a.dx/div, a.dy/div, a.aspect);

        if(flip) flip_image(sized_m);
        d.y.vals[i] = sized_m.data;

        free_image(orig);
        free_image(mask);

        /*
           image rgb = mask_to_rgb(sized_m, classes);
           show_image(rgb, "part");
           show_image(sized, "orig");
           cvWaitKey(0);
           free_image(rgb);
         */
    }
    free(random_paths);
    return d;
}

data load_data_iseg(int n, char **paths, int m, int w, int h, int classes, int boxes, int div, int min, int max, float angle, float aspect, float hue, float saturation, float exposure)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    d.y = make_matrix(n, (((w/div)*(h/div))+1)*boxes);

    for(i = 0; i < n; ++i){
        image orig = load_image_color(random_paths[i], 0, 0);
        augment_args a = random_augment_args(orig, angle, aspect, min, max, w, h);
        image sized = rotate_crop_image(orig, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);

        int flip = rand()%2;
        if(flip) flip_image(sized);
        random_distort_image(sized, hue, saturation, exposure);
        d.X.vals[i] = sized.data;
        //show_image(sized, "image");

        fill_truth_iseg(random_paths[i], boxes, d.y.vals[i], classes, orig.w, orig.h, a, flip, w/div, h/div);

        free_image(orig);

        /*
           image rgb = mask_to_rgb(sized_m, classes);
           show_image(rgb, "part");
           show_image(sized, "orig");
           cvWaitKey(0);
           free_image(rgb);
         */
    }
    free(random_paths);
    return d;
}

data load_data_mask(int n, char **paths, int m, int w, int h, int classes, int boxes, int coords, int min, int max, float angle, float aspect, float hue, float saturation, float exposure)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    d.y = make_matrix(n, (coords+1)*boxes);

    for(i = 0; i < n; ++i){
        image orig = load_image_color(random_paths[i], 0, 0);
        augment_args a = random_augment_args(orig, angle, aspect, min, max, w, h);
        image sized = rotate_crop_image(orig, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);

        int flip = rand()%2;
        if(flip) flip_image(sized);
        random_distort_image(sized, hue, saturation, exposure);
        d.X.vals[i] = sized.data;
        //show_image(sized, "image");

        fill_truth_mask(random_paths[i], boxes, d.y.vals[i], classes, orig.w, orig.h, a, flip, 14, 14);

        free_image(orig);

        /*
           image rgb = mask_to_rgb(sized_m, classes);
           show_image(rgb, "part");
           show_image(sized, "orig");
           cvWaitKey(0);
           free_image(rgb);
         */
    }
    free(random_paths);
    return d;
}

data load_data_region(int n, char **paths, int m, int w, int h, int size, int classes, float jitter, float hue, float saturation, float exposure)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;


    int k = size*size*(5+classes);
    d.y = make_matrix(n, k);
    for(i = 0; i < n; ++i){
        image orig = load_image_color(random_paths[i], 0, 0);

        int oh = orig.h;
        int ow = orig.w;

        int dw = (ow*jitter);
        int dh = (oh*jitter);

        int pleft  = rand_uniform(-dw, dw);
        int pright = rand_uniform(-dw, dw);
        int ptop   = rand_uniform(-dh, dh);
        int pbot   = rand_uniform(-dh, dh);

        int swidth =  ow - pleft - pright;
        int sheight = oh - ptop - pbot;

        float sx = (float)swidth  / ow;
        float sy = (float)sheight / oh;

        int flip = rand()%2;
        image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

        float dx = ((float)pleft/ow)/sx;
        float dy = ((float)ptop /oh)/sy;

        image sized = resize_image(cropped, w, h);
        if(flip) flip_image(sized);
        random_distort_image(sized, hue, saturation, exposure);
        d.X.vals[i] = sized.data;

        fill_truth_region(random_paths[i], d.y.vals[i], classes, size, flip, dx, dy, 1./sx, 1./sy);

        free_image(orig);
        free_image(cropped);
    }
    free(random_paths);
    return d;
}

data load_data_compare(int n, char **paths, int m, int classes, int w, int h)
{
    if(m) paths = get_random_paths(paths, 2*n, m);
    int i,j;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*6;

    int k = 2*(classes);
    d.y = make_matrix(n, k);
    for(i = 0; i < n; ++i){
        image im1 = load_image_color(paths[i*2],   w, h);
        image im2 = load_image_color(paths[i*2+1], w, h);

        d.X.vals[i] = calloc(d.X.cols, sizeof(float));
        memcpy(d.X.vals[i],         im1.data, h*w*3*sizeof(float));
        memcpy(d.X.vals[i] + h*w*3, im2.data, h*w*3*sizeof(float));

        int id;
        float iou;

        char imlabel1[4096];
        char imlabel2[4096];
        find_replace(paths[i*2],   "imgs", "labels", imlabel1);
        find_replace(imlabel1, "jpg", "txt", imlabel1);
        FILE *fp1 = fopen(imlabel1, "r");

        while(fscanf(fp1, "%d %f", &id, &iou) == 2){
            if (d.y.vals[i][2*id] < iou) d.y.vals[i][2*id] = iou;
        }

        find_replace(paths[i*2+1], "imgs", "labels", imlabel2);
        find_replace(imlabel2, "jpg", "txt", imlabel2);
        FILE *fp2 = fopen(imlabel2, "r");

        while(fscanf(fp2, "%d %f", &id, &iou) == 2){
            if (d.y.vals[i][2*id + 1] < iou) d.y.vals[i][2*id + 1] = iou;
        }

        for (j = 0; j < classes; ++j){
            if (d.y.vals[i][2*j] > .5 &&  d.y.vals[i][2*j+1] < .5){
                d.y.vals[i][2*j] = 1;
                d.y.vals[i][2*j+1] = 0;
            } else if (d.y.vals[i][2*j] < .5 &&  d.y.vals[i][2*j+1] > .5){
                d.y.vals[i][2*j] = 0;
                d.y.vals[i][2*j+1] = 1;
            } else {
                d.y.vals[i][2*j]   = SECRET_NUM;
                d.y.vals[i][2*j+1] = SECRET_NUM;
            }
        }
        fclose(fp1);
        fclose(fp2);

        free_image(im1);
        free_image(im2);
    }
    if(m) free(paths);
    return d;
}

data load_data_swag(char **paths, int n, int classes, float jitter)
{
    int index = rand()%n;
    char *random_path = paths[index];

    image orig = load_image_color(random_path, 0, 0);
    int h = orig.h;
    int w = orig.w;

    data d = {0};
    d.shallow = 0;
    d.w = w;
    d.h = h;

    d.X.rows = 1;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    int k = (4+classes)*90;
    d.y = make_matrix(1, k);

    int dw = w*jitter;
    int dh = h*jitter;

    int pleft  = rand_uniform(-dw, dw);
    int pright = rand_uniform(-dw, dw);
    int ptop   = rand_uniform(-dh, dh);
    int pbot   = rand_uniform(-dh, dh);

    int swidth =  w - pleft - pright;
    int sheight = h - ptop - pbot;

    float sx = (float)swidth  / w;
    float sy = (float)sheight / h;

    int flip = rand()%2;
    image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

    float dx = ((float)pleft/w)/sx;
    float dy = ((float)ptop /h)/sy;

    image sized = resize_image(cropped, w, h);
    if(flip) flip_image(sized);
    d.X.vals[0] = sized.data;

    fill_truth_swag(random_path, d.y.vals[0], classes, flip, dx, dy, 1./sx, 1./sy);

    free_image(orig);
    free_image(cropped);

    return d;
}

unsigned int random_gen()
{
    unsigned int rnd = 0;
#ifdef WIN32
    rand_s(&rnd);
#else   // WIN32
    rnd = rand();
#if (RAND_MAX < 65536)
        rnd = rand()*(RAND_MAX + 1) + rnd;
#endif  //(RAND_MAX < 65536)
#endif  // WIN32
    return rnd;
}

float random_float()
{
    unsigned int rnd = 0;
#ifdef WIN32
    rand_s(&rnd);
    return ((float)rnd / (float)UINT_MAX);
#else   // WIN32

    rnd = rand();
#if (RAND_MAX < 65536)
    rnd = rand()*(RAND_MAX + 1) + rnd;
    return((float)rnd / (float)(RAND_MAX*RAND_MAX));
#endif  //(RAND_MAX < 65536)
    return ((float)rnd / (float)RAND_MAX);

#endif  // WIN32
}

float rand_uniform_strong(float min, float max)
{
    if (max < min) {
        float swap = min;
        min = max;
        max = swap;
    }
    return (random_float() * (max - min)) + min;
}

// Marsaglia's xorshf96 generator: period 2^96-1
unsigned int random_gen_fast(void)
{
    unsigned int t;
    static unsigned int x = 123456789, y = 362436069, z = 521288629;
    x ^= x << 16;
    x ^= x >> 5;
    x ^= x << 1;

    t = x;
    x = y;
    y = z;
    z = t ^ x ^ y;

    return z;
}

float rand_precalc_random(float min, float max, float random_part)
{
    if (max < min) {
        float swap = min;
        min = max;
        max = swap;
    }
    return (random_part * (max - min)) + min;
}

char **get_sequential_paths(char **paths, int n, int m, int mini_batch, int augment_speed)
{
    int speed = rand_int(1, augment_speed);
    if (speed < 1) speed = 1;
    char** sequentia_paths = (char**)calloc(n, sizeof(char*));
    int i;
    pthread_mutex_lock(&mutex);
    //printf("n = %d, mini_batch = %d \n", n, mini_batch);
    unsigned int *start_time_indexes = (unsigned int *)calloc(mini_batch, sizeof(unsigned int));
    for (i = 0; i < mini_batch; ++i) {
        start_time_indexes[i] = random_gen() % m;
        //printf(" start_time_indexes[i] = %u, ", start_time_indexes[i]);
    }

    for (i = 0; i < n; ++i) {
        do {
            int time_line_index = i % mini_batch;
            unsigned int index = start_time_indexes[time_line_index] % m;
            start_time_indexes[time_line_index] += speed;

            //int index = random_gen() % m;
            sequentia_paths[i] = paths[index];
            //if(i == 0) printf("%s\n", paths[index]);
            //printf(" index = %u - grp: %s \n", index, paths[index]);
            if (strlen(sequentia_paths[i]) <= 4) printf(" Very small path to the image: %s \n", sequentia_paths[i]);
        } while (strlen(sequentia_paths[i]) == 0);
    }
    free(start_time_indexes);
    pthread_mutex_unlock(&mutex);
    return sequentia_paths;
}

float random_float_fast()
{
    return ((float)random_gen_fast() / (float)UINT_MAX);
}

static box float_to_box_stride(float *f, int stride)
{
    box b = { 0 };
    b.x = f[0];
    b.y = f[1 * stride];
    b.w = f[2 * stride];
    b.h = f[3 * stride];
    return b;
}

void blend_truth(float *new_truth, int boxes, float *old_truth)
{
    const int t_size = 4 + 1;
    int count_new_truth = 0;
    int t;
    for (t = 0; t < boxes; ++t) {
        float x = new_truth[t*(4 + 1)];
        if (!x) break;
        count_new_truth++;

    }
    for (t = count_new_truth; t < boxes; ++t) {
        float *new_truth_ptr = new_truth + t*t_size;
        float *old_truth_ptr = old_truth + (t - count_new_truth)*t_size;
        float x = old_truth_ptr[0];
        if (!x) break;

        new_truth_ptr[0] = old_truth_ptr[0];
        new_truth_ptr[1] = old_truth_ptr[1];
        new_truth_ptr[2] = old_truth_ptr[2];
        new_truth_ptr[3] = old_truth_ptr[3];
        new_truth_ptr[4] = old_truth_ptr[4];
    }
    //printf("\n was %d bboxes, now %d bboxes \n", count_new_truth, t);
}


#ifdef OPENCV

//#include "http_stream.h"
//#include <assert.h>
//typedef void* mat_cv;

#define PJREDDIE
#ifdef PJREDDIE
data load_data_detection(int n, char **paths, int m, int w, int h, int c, int boxes, int classes, int use_flip, int use_blur, int use_mixup,
    float jitter, float hue, float saturation, float exposure, int mini_batch, int track, int augment_speed, int letter_box, int show_imgs)
//data load_data_detection(int n, char **paths, int m, int w, int h, int boxes, int classes, float jitter, float hue, float saturation, float exposure)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    d.y = make_matrix(n, 5*boxes);
    for(i = 0; i < n; ++i){
        image orig = load_image_color(random_paths[i], 0, 0);
        image sized = make_image(w, h, orig.c);
        fill_image(sized, .5);

        float dw = jitter * orig.w;
        float dh = jitter * orig.h;

        float new_ar = (orig.w + rand_uniform(-dw, dw)) / (orig.h + rand_uniform(-dh, dh));
        //float scale = rand_uniform(.25, 2);
        float scale = 1;

        float nw, nh;

        if(new_ar < 1){
            nh = scale * h;
            nw = nh * new_ar;
        } else {
            nw = scale * w;
            nh = nw / new_ar;
        }

        float dx = rand_uniform(0, w - nw);
        float dy = rand_uniform(0, h - nh);

        place_image(orig, nw, nh, dx, dy, sized);

        random_distort_image(sized, hue, saturation, exposure);

        int flip = rand()%2;
        if(flip) flip_image(sized);
        d.X.vals[i] = sized.data;


        fill_truth_detection_pjreddie(random_paths[i], boxes, d.y.vals[i], classes, flip, -dx/w, -dy/h, nw/w, nh/h);

        free_image(orig);
    }
    free(random_paths);
    return d;
}
#else   // AlexeyAB
data load_data_detection(int n, char **paths, int m, int w, int h, int c, int boxes, int classes, int use_flip, int use_blur, int use_mixup,
    float jitter, float hue, float saturation, float exposure, int mini_batch, int track, int augment_speed, int letter_box, int show_imgs)
{
    const int random_index = random_gen();
    c = c ? c : 3;

    assert(use_mixup != 2);
    if (use_mixup == 3 && letter_box) {
        printf("\n Combination: letter_box=1 & mosaic=1 - isn't supported, use only 1 of these parameters \n");
        exit(0);
    }
    if (random_gen() % 2 == 0) use_mixup = 0;
    int i;

    int *cut_x = NULL, *cut_y = NULL;
    if (use_mixup == 3) {
        cut_x = (int*)calloc(n, sizeof(int));
        cut_y = (int*)calloc(n, sizeof(int));
        const float min_offset = 0.2; // 20%
        for (i = 0; i < n; ++i) {
            cut_x[i] = rand_int(w*min_offset, w*(1 - min_offset));
            cut_y[i] = rand_int(h*min_offset, h*(1 - min_offset));
        }
    }

    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = (float**)calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*c;

    float r1 = 0, r2 = 0, r3 = 0, r4 = 0, r_scale = 0;
    float dhue = 0, dsat = 0, dexp = 0, flip = 0, blur = 0;
    int augmentation_calculated = 0;

    d.y = make_matrix(n, 5*boxes);
    int i_mixup = 0;
    for (i_mixup = 0; i_mixup <= use_mixup; i_mixup++) {
        if (i_mixup) augmentation_calculated = 0;   // recalculate augmentation for the 2nd sequence if(track==1)

        char **random_paths;
        if (track) random_paths = get_sequential_paths(paths, n, m, mini_batch, augment_speed);
        else random_paths = get_random_paths(paths, n, m);

        for (i = 0; i < n; ++i) {
            float *truth = (float*)calloc(5 * boxes, sizeof(float));
            const char *filename = random_paths[i];

            int flag = (c >= 3);
            mat_cv *src;
            src = load_image_mat_cv(filename, flag);
            if (src == NULL) {
                if (check_mistakes) {
                    printf("\n Error in load_data_detection() - OpenCV \n");
                    getchar();
                }
                continue;
            }

            int oh = get_height_mat(src);
            int ow = get_width_mat(src);

            int dw = (ow*jitter);
            int dh = (oh*jitter);

            if (!augmentation_calculated || !track)
            {
                augmentation_calculated = 1;
                r1 = random_float();
                r2 = random_float();
                r3 = random_float();
                r4 = random_float();

                r_scale = random_float();

                dhue = rand_uniform_strong(-hue, hue);
                dsat = rand_scale(saturation);
                dexp = rand_scale(exposure);

                flip = use_flip ? random_gen() % 2 : 0;

                if (use_blur) {
                    int tmp_blur = rand_int(0, 2);  // 0 - disable, 1 - blur background, 2 - blur the whole image
                    if (tmp_blur == 0) blur = 0;
                    else if (tmp_blur == 1) blur = 1;
                    else blur = use_blur;
                }
            }

            int pleft = rand_precalc_random(-dw, dw, r1);
            int pright = rand_precalc_random(-dw, dw, r2);
            int ptop = rand_precalc_random(-dh, dh, r3);
            int pbot = rand_precalc_random(-dh, dh, r4);
            //printf("\n pleft = %d, pright = %d, ptop = %d, pbot = %d, ow = %d, oh = %d \n", pleft, pright, ptop, pbot, ow, oh);

            //float scale = rand_precalc_random(.25, 2, r_scale); // unused currently

            if (letter_box)
            {
                float img_ar = (float)ow / (float)oh;
                float net_ar = (float)w / (float)h;
                float result_ar = img_ar / net_ar;
                //printf(" ow = %d, oh = %d, w = %d, h = %d, img_ar = %f, net_ar = %f, result_ar = %f \n", ow, oh, w, h, img_ar, net_ar, result_ar);
                if (result_ar > 1)  // sheight - should be increased
                {
                    float oh_tmp = ow / net_ar;
                    float delta_h = (oh_tmp - oh)/2;
                    ptop = ptop - delta_h;
                    pbot = pbot - delta_h;
                    //printf(" result_ar = %f, oh_tmp = %f, delta_h = %d, ptop = %f, pbot = %f \n", result_ar, oh_tmp, delta_h, ptop, pbot);
                }
                else  // swidth - should be increased
                {
                    float ow_tmp = oh * net_ar;
                    float delta_w = (ow_tmp - ow)/2;
                    pleft = pleft - delta_w;
                    pright = pright - delta_w;
                    //printf(" result_ar = %f, ow_tmp = %f, delta_w = %d, pleft = %f, pright = %f \n", result_ar, ow_tmp, delta_w, pleft, pright);
                }
            }

            int swidth = ow - pleft - pright;
            int sheight = oh - ptop - pbot;

            float sx = (float)swidth / ow;
            float sy = (float)sheight / oh;

            float dx = ((float)pleft / ow) / sx;
            float dy = ((float)ptop / oh) / sy;


            int min_w_h = fill_truth_detection(filename, boxes, truth, classes, flip, dx, dy, 1. / sx, 1. / sy, w, h);

            if ((min_w_h / 8) < blur && blur > 1) blur = min_w_h / 8;   // disable blur if one of the objects is too small

            image ai = image_data_augmentation(src, w, h, pleft, ptop, swidth, sheight, flip, dhue, dsat, dexp,
                blur, boxes, truth);

            if (use_mixup == 0) {
                d.X.vals[i] = ai.data;
                memcpy(d.y.vals[i], truth, 5 * boxes * sizeof(float));
            }
            else if (use_mixup == 1) {
                if (i_mixup == 0) {
                    d.X.vals[i] = ai.data;
                    memcpy(d.y.vals[i], truth, 5 * boxes * sizeof(float));
                }
                else if (i_mixup == 1) {
                    image old_img = make_empty_image(w, h, c);
                    old_img.data = d.X.vals[i];
                    //show_image(ai, "new");
                    //show_image(old_img, "old");
                    //wait_until_press_key_cv();
                    blend_images_cv(ai, 0.5, old_img, 0.5);
                    blend_truth(d.y.vals[i], boxes, truth);
                    free_image(old_img);
                    d.X.vals[i] = ai.data;
                }
            }
            else if (use_mixup == 3) {
                if (i_mixup == 0) {
                    image tmp_img = make_image(w, h, c);
                    d.X.vals[i] = tmp_img.data;
                }

                if (flip) {
                    int tmp = pleft;
                    pleft = pright;
                    pright = tmp;
                }

                const int left_shift = min_val_cmp(cut_x[i], max_val_cmp(0, (-pleft*w / ow)));
                const int top_shift = min_val_cmp(cut_y[i], max_val_cmp(0, (-ptop*h / oh)));

                const int right_shift = min_val_cmp((w - cut_x[i]), max_val_cmp(0, (-pright*w / ow)));
                const int bot_shift = min_val_cmp(h - cut_y[i], max_val_cmp(0, (-pbot*h / oh)));


                int k, x, y;
                for (k = 0; k < c; ++k) {
                    for (y = 0; y < h; ++y) {
                        int j = y*w + k*w*h;
                        if (i_mixup == 0 && y < cut_y[i]) {
                            int j_src = (w - cut_x[i] - right_shift) + (y + h - cut_y[i] - bot_shift)*w + k*w*h;
                            memcpy(&d.X.vals[i][j + 0], &ai.data[j_src], cut_x[i] * sizeof(float));
                        }
                        if (i_mixup == 1 && y < cut_y[i]) {
                            int j_src = left_shift + (y + h - cut_y[i] - bot_shift)*w + k*w*h;
                            memcpy(&d.X.vals[i][j + cut_x[i]], &ai.data[j_src], (w-cut_x[i]) * sizeof(float));
                        }
                        if (i_mixup == 2 && y >= cut_y[i]) {
                            int j_src = (w - cut_x[i] - right_shift) + (top_shift + y - cut_y[i])*w + k*w*h;
                            memcpy(&d.X.vals[i][j + 0], &ai.data[j_src], cut_x[i] * sizeof(float));
                        }
                        if (i_mixup == 3 && y >= cut_y[i]) {
                            int j_src = left_shift + (top_shift + y - cut_y[i])*w + k*w*h;
                            memcpy(&d.X.vals[i][j + cut_x[i]], &ai.data[j_src], (w - cut_x[i]) * sizeof(float));
                        }
                    }
                }

                blend_truth_mosaic(d.y.vals[i], boxes, truth, w, h, cut_x[i], cut_y[i], i_mixup, left_shift, right_shift, top_shift, bot_shift);

                free_image(ai);
                ai.data = d.X.vals[i];
            }


            if (show_imgs && i_mixup == use_mixup)   // delete i_mixup
            {
                image tmp_ai = copy_image(ai);
                char buff[1000];
                //sprintf(buff, "aug_%d_%d_%s_%d", random_index, i, basecfg((char*)filename), random_gen());
                sprintf(buff, "aug_%d_%d_%d", random_index, i, random_gen());
                int t;
                for (t = 0; t < boxes; ++t) {
                    box b = float_to_box_stride(d.y.vals[i] + t*(4 + 1), 1);
                    if (!b.x) break;
                    int left = (b.x - b.w / 2.)*ai.w;
                    int right = (b.x + b.w / 2.)*ai.w;
                    int top = (b.y - b.h / 2.)*ai.h;
                    int bot = (b.y + b.h / 2.)*ai.h;
                    draw_box_width(tmp_ai, left, top, right, bot, 1, 150, 100, 50); // 3 channels RGB
                }

                save_image(tmp_ai, buff);
                if (show_imgs == 1) {
                    //char buff_src[1000];
                    //sprintf(buff_src, "src_%d_%d_%s_%d", random_index, i, basecfg((char*)filename), random_gen());
                    //show_image_mat(src, buff_src);
                    show_image(tmp_ai, buff, 1);
                    wait_until_press_key_cv();
                }
                printf("\nYou use flag -show_imgs, so will be saved aug_...jpg images. Click on window and press ESC button \n");
                free_image(tmp_ai);
            }

            release_mat(&src);
            free(truth);
        }
        if (random_paths) free(random_paths);
    }


    return d;
}
#endif // PJREDDIE

#else    // OPENCV
#include <assert.h>
void blend_images(image new_img, float alpha, image old_img, float beta)
{
    int data_size = new_img.w * new_img.h * new_img.c;
    int i;
    #pragma omp parallel for
    for (i = 0; i < data_size; ++i)
        new_img.data[i] = new_img.data[i] * alpha + old_img.data[i] * beta;
}

data load_data_detection(int n, char **paths, int m, int w, int h, int c, int boxes, int classes, int use_flip, int use_blur, int use_mixup, float jitter,
    float hue, float saturation, float exposure, int mini_batch, int track, int augment_speed, int letter_box, int show_imgs)
{
    const int random_index = random_gen();
    c = c ? c : 3;
    char **random_paths;
    char **mixup_random_paths = NULL;
    if(track) random_paths = get_sequential_paths(paths, n, m, mini_batch, augment_speed);
    else random_paths = get_random_paths(paths, n, m);

    assert(use_mixup < 2);
    int mixup = use_mixup ? random_gen() % 2 : 0;
    //printf("\n mixup = %d \n", mixup);
    if (mixup) {
        if (track) mixup_random_paths = get_sequential_paths(paths, n, m, mini_batch, augment_speed);
        else mixup_random_paths = get_random_paths(paths, n, m);
    }

    int i;
    data d = { 0 };
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = (float**)calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*c;

    float r1 = 0, r2 = 0, r3 = 0, r4 = 0;
 //   float r1 = 0, r2 = 0, r3 = 0, r4 = 0, r_scale;
    float dhue = 0, dsat = 0, dexp = 0, flip = 0;
    int augmentation_calculated = 0;

    d.y = make_matrix(n, 5 * boxes);
    int i_mixup = 0;
    for (i_mixup = 0; i_mixup <= mixup; i_mixup++) {
        if (i_mixup) augmentation_calculated = 0;
        for (i = 0; i < n; ++i) {
            float *truth = (float*)calloc(5 * boxes, sizeof(float));
            char *filename = (i_mixup) ? mixup_random_paths[i] : random_paths[i];

            image orig = load_image(filename, 0, 0, c);

            int oh = orig.h;
            int ow = orig.w;

            int dw = (ow*jitter);
            int dh = (oh*jitter);

            if (!augmentation_calculated || !track)
            {
                augmentation_calculated = 1;
                r1 = random_float();
                r2 = random_float();
                r3 = random_float();
                r4 = random_float();

       //         r_scale = random_float();

                dhue = rand_uniform_strong(-hue, hue);
                dsat = rand_scale(saturation);
                dexp = rand_scale(exposure);

                flip = use_flip ? random_gen() % 2 : 0;
            }

            int pleft = rand_precalc_random(-dw, dw, r1);
            int pright = rand_precalc_random(-dw, dw, r2);
            int ptop = rand_precalc_random(-dh, dh, r3);
            int pbot = rand_precalc_random(-dh, dh, r4);

       //     float scale = rand_precalc_random(.25, 2, r_scale); // unused currently

            if (letter_box)
            {
                float img_ar = (float)ow / (float)oh;
                float net_ar = (float)w / (float)h;
                float result_ar = img_ar / net_ar;
                //printf(" ow = %d, oh = %d, w = %d, h = %d, img_ar = %f, net_ar = %f, result_ar = %f \n", ow, oh, w, h, img_ar, net_ar, result_ar);
                if (result_ar > 1)  // sheight - should be increased
                {
                    float oh_tmp = ow / net_ar;
                    float delta_h = (oh_tmp - oh) / 2;
                    ptop = ptop - delta_h;
                    pbot = pbot - delta_h;
                    //printf(" result_ar = %f, oh_tmp = %f, delta_h = %d, ptop = %f, pbot = %f \n", result_ar, oh_tmp, delta_h, ptop, pbot);
                }
                else  // swidth - should be increased
                {
                    float ow_tmp = oh * net_ar;
                    float delta_w = (ow_tmp - ow) / 2;
                    pleft = pleft - delta_w;
                    pright = pright - delta_w;
                    //printf(" result_ar = %f, ow_tmp = %f, delta_w = %d, pleft = %f, pright = %f \n", result_ar, ow_tmp, delta_w, pleft, pright);
                }
            }

            int swidth = ow - pleft - pright;
            int sheight = oh - ptop - pbot;

            float sx = (float)swidth / ow;
            float sy = (float)sheight / oh;

            image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

            float dx = ((float)pleft / ow) / sx;
            float dy = ((float)ptop / oh) / sy;

            image sized = resize_image(cropped, w, h);
            if (flip) flip_image(sized);
            distort_image(sized, dhue, dsat, dexp);
            //random_distort_image(sized, hue, saturation, exposure);

            fill_truth_detection(filename, boxes, truth, classes, flip, dx, dy, 1. / sx, 1. / sy, w, h);

            if (i_mixup) {
                image old_img = sized;
                old_img.data = d.X.vals[i];
                //show_image(sized, "new");
                //show_image(old_img, "old");
                //wait_until_press_key_cv();
                blend_images(sized, 0.5, old_img, 0.5);
                blend_truth(truth, boxes, d.y.vals[i]);
                free_image(old_img);
            }

            d.X.vals[i] = sized.data;
            memcpy(d.y.vals[i], truth, 5 * boxes * sizeof(float));

            if (show_imgs)// && i_mixup)
            {
                char buff[1000];
                sprintf(buff, "aug_%d_%d_%s_%d", random_index, i, basecfg(filename), random_gen());

                int t;
                for (t = 0; t < boxes; ++t) {
                    box b = float_to_box_stride(d.y.vals[i] + t*(4 + 1), 1);
                    if (!b.x) break;
                    int left = (b.x - b.w / 2.)*sized.w;
                    int right = (b.x + b.w / 2.)*sized.w;
                    int top = (b.y - b.h / 2.)*sized.h;
                    int bot = (b.y + b.h / 2.)*sized.h;
                    draw_box_width(sized, left, top, right, bot, 1, 150, 100, 50); // 3 channels RGB
                }

                save_image(sized, buff);
                if (show_imgs == 1) {
                    show_image(sized, buff, 1);
            //        wait_until_press_key_cv();
                }
                printf("\nYou use flag -show_imgs, so will be saved aug_...jpg images. Press Enter: \n");
                //getchar();
            }

            free_image(orig);
            free_image(cropped);
            free(truth);
        }
    }
    free(random_paths);
    if (mixup_random_paths) free(mixup_random_paths);
    return d;
}
#endif    // OPENCV

/*data load_data_detection(int n, char **paths, int m, int w, int h, int boxes, int classes, float jitter, float hue, float saturation, float exposure)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    d.y = make_matrix(n, 5*boxes);
    for(i = 0; i < n; ++i){
        image orig = load_image_color(random_paths[i], 0, 0);
        image sized = make_image(w, h, orig.c);
        fill_image(sized, .5);

        float dw = jitter * orig.w;
        float dh = jitter * orig.h;

        float new_ar = (orig.w + rand_uniform(-dw, dw)) / (orig.h + rand_uniform(-dh, dh));
        //float scale = rand_uniform(.25, 2);
        float scale = 1;

        float nw, nh;

        if(new_ar < 1){
            nh = scale * h;
            nw = nh * new_ar;
        } else {
            nw = scale * w;
            nh = nw / new_ar;
        }

        float dx = rand_uniform(0, w - nw);
        float dy = rand_uniform(0, h - nh);

        place_image(orig, nw, nh, dx, dy, sized);

        random_distort_image(sized, hue, saturation, exposure);

        int flip = rand()%2;
        if(flip) flip_image(sized);
        d.X.vals[i] = sized.data;


        fill_truth_detection(random_paths[i], boxes, d.y.vals[i], classes, flip, -dx/w, -dy/h, nw/w, nh/h);

        free_image(orig);
    }
    free(random_paths);
    return d;
}*/

void *load_thread(void *ptr)
{
    //printf("Loading data: %d\n", rand());
    load_args a = *(struct load_args*)ptr;
    if(a.exposure == 0) a.exposure = 1;
    if(a.saturation == 0) a.saturation = 1;
    if(a.aspect == 0) a.aspect = 1;

    if (a.type == OLD_CLASSIFICATION_DATA){
        *a.d = load_data_old(a.paths, a.n, a.m, a.labels, a.classes, a.w, a.h);
    } else if (a.type == REGRESSION_DATA){
        *a.d = load_data_regression(a.paths, a.n, a.m, a.classes, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
    } else if (a.type == CLASSIFICATION_DATA){
        *a.d = load_data_augment(a.paths, a.n, a.m, a.labels, a.classes, a.hierarchy, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure, a.center);
    } else if (a.type == SUPER_DATA){
        *a.d = load_data_super(a.paths, a.n, a.m, a.w, a.h, a.scale);
    } else if (a.type == WRITING_DATA){
        *a.d = load_data_writing(a.paths, a.n, a.m, a.w, a.h, a.out_w, a.out_h);
    } else if (a.type == ISEG_DATA){
        *a.d = load_data_iseg(a.n, a.paths, a.m, a.w, a.h, a.classes, a.num_boxes, a.scale, a.min, a.max, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
    } else if (a.type == INSTANCE_DATA){
        *a.d = load_data_mask(a.n, a.paths, a.m, a.w, a.h, a.classes, a.num_boxes, a.coords, a.min, a.max, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
    } else if (a.type == SEGMENTATION_DATA){
        *a.d = load_data_seg(a.n, a.paths, a.m, a.w, a.h, a.classes, a.min, a.max, a.angle, a.aspect, a.hue, a.saturation, a.exposure, a.scale);
    } else if (a.type == REGION_DATA){
        *a.d = load_data_region(a.n, a.paths, a.m, a.w, a.h, a.num_boxes, a.classes, a.jitter, a.hue, a.saturation, a.exposure);
    } else if (a.type == DETECTION_DATA){
        //*a.d = load_data_detection(a.n, a.paths, a.m, a.w, a.h, a.num_boxes, a.classes, a.jitter, a.hue, a.saturation, a.exposure);
        *a.d = load_data_detection(a.n, a.paths, a.m, a.w, a.h, a.c, a.num_boxes, a.classes, a.flip, a.blur, a.mixup, a.jitter,
            a.hue, a.saturation, a.exposure, a.mini_batch, a.track, a.augment_speed, a.letter_box, a.show_imgs);
    } else if (a.type == SWAG_DATA){
        *a.d = load_data_swag(a.paths, a.n, a.classes, a.jitter);
    } else if (a.type == COMPARE_DATA){
        *a.d = load_data_compare(a.n, a.paths, a.m, a.classes, a.w, a.h);
    } else if (a.type == IMAGE_DATA){
        *(a.im) = load_image_color(a.path, 0, 0);
        *(a.resized) = resize_image(*(a.im), a.w, a.h);
    } else if (a.type == LETTERBOX_DATA){
        *(a.im) = load_image_color(a.path, 0, 0);
        *(a.resized) = letterbox_image(*(a.im), a.w, a.h);
    } else if (a.type == TAG_DATA){
        *a.d = load_data_tag(a.paths, a.n, a.m, a.classes, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
    }
    free(ptr);
    return 0;
}

pthread_t load_data_in_thread(load_args args)
{
    pthread_t thread;
    struct load_args *ptr = calloc(1, sizeof(struct load_args));
    *ptr = args;
    if(pthread_create(&thread, 0, load_thread, ptr)) error("Thread creation failed");
    return thread;
}

void *load_threads(void *ptr)
{
    int i;
    load_args args = *(load_args *)ptr;
    if (args.threads == 0) args.threads = 1;
    data *out = args.d;
    int total = args.n;
    free(ptr);
    data *buffers = calloc(args.threads, sizeof(data));
    pthread_t *threads = calloc(args.threads, sizeof(pthread_t));
    for(i = 0; i < args.threads; ++i){
        args.d = buffers + i;
        args.n = (i+1) * total/args.threads - i * total/args.threads;
        threads[i] = load_data_in_thread(args);
    }
    for(i = 0; i < args.threads; ++i){
        pthread_join(threads[i], 0);
    }
    *out = concat_datas(buffers, args.threads);
    out->shallow = 0;
    for(i = 0; i < args.threads; ++i){
        buffers[i].shallow = 1;
        free_data(buffers[i]);
    }
    free(buffers);
    free(threads);
    return 0;
}

void load_data_blocking(load_args args)
{
    struct load_args *ptr = calloc(1, sizeof(struct load_args));
    *ptr = args;
    load_thread(ptr);
}

pthread_t load_data(load_args args)
{
    pthread_t thread;
    struct load_args *ptr = calloc(1, sizeof(struct load_args));
    *ptr = args;
    if(pthread_create(&thread, 0, load_threads, ptr)) error("Thread creation failed");
    return thread;
}

data load_data_writing(char **paths, int n, int m, int w, int h, int out_w, int out_h)
{
    if(m) paths = get_random_paths(paths, n, m);
    char **replace_paths = find_replace_paths(paths, n, ".png", "-label.png");
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = load_image_paths_gray(replace_paths, n, out_w, out_h);
    if(m) free(paths);
    int i;
    for(i = 0; i < n; ++i) free(replace_paths[i]);
    free(replace_paths);
    return d;
}

data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = load_labels_paths(paths, n, labels, k, 0);
    if(m) free(paths);
    return d;
}

/*
   data load_data_study(char **paths, int n, int m, char **labels, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
   {
   data d = {0};
   d.indexes = calloc(n, sizeof(int));
   if(m) paths = get_random_paths_indexes(paths, n, m, d.indexes);
   d.shallow = 0;
   d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure);
   d.y = load_labels_paths(paths, n, labels, k);
   if(m) free(paths);
   return d;
   }
 */

data load_data_super(char **paths, int n, int m, int w, int h, int scale)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;

    int i;
    d.X.rows = n;
    d.X.vals = calloc(n, sizeof(float*));
    d.X.cols = w*h*3;

    d.y.rows = n;
    d.y.vals = calloc(n, sizeof(float*));
    d.y.cols = w*scale * h*scale * 3;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], 0, 0);
        image crop = random_crop_image(im, w*scale, h*scale);
        int flip = rand()%2;
        if (flip) flip_image(crop);
        image resize = resize_image(crop, w, h);
        d.X.vals[i] = resize.data;
        d.y.vals[i] = crop.data;
        free_image(im);
    }

    if(m) free(paths);
    return d;
}

data load_data_regression(char **paths, int n, int m, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure, 0);
    d.y = load_regression_labels_paths(paths, n, k);
    if(m) free(paths);
    return d;
}

data select_data(data *orig, int *inds)
{
    data d = {0};
    d.shallow = 1;
    d.w = orig[0].w;
    d.h = orig[0].h;

    d.X.rows = orig[0].X.rows;
    d.y.rows = orig[0].X.rows;

    d.X.cols = orig[0].X.cols;
    d.y.cols = orig[0].y.cols;

    d.X.vals = calloc(orig[0].X.rows, sizeof(float *));
    d.y.vals = calloc(orig[0].y.rows, sizeof(float *));
    int i;
    for(i = 0; i < d.X.rows; ++i){
        d.X.vals[i] = orig[inds[i]].X.vals[i];
        d.y.vals[i] = orig[inds[i]].y.vals[i];
    }
    return d;
}

data *tile_data(data orig, int divs, int size)
{
    data *ds = calloc(divs*divs, sizeof(data));
    int i, j;
#pragma omp parallel for
    for(i = 0; i < divs*divs; ++i){
        data d;
        d.shallow = 0;
        d.w = orig.w/divs * size;
        d.h = orig.h/divs * size;
        d.X.rows = orig.X.rows;
        d.X.cols = d.w*d.h*3;
        d.X.vals = calloc(d.X.rows, sizeof(float*));

        d.y = copy_matrix(orig.y);
#pragma omp parallel for
        for(j = 0; j < orig.X.rows; ++j){
            int x = (i%divs) * orig.w / divs - (d.w - orig.w/divs)/2;
            int y = (i/divs) * orig.h / divs - (d.h - orig.h/divs)/2;
            image im = float_to_image(orig.w, orig.h, 3, orig.X.vals[j]);
            d.X.vals[j] = crop_image(im, x, y, d.w, d.h).data;
        }
        ds[i] = d;
    }
    return ds;
}

data resize_data(data orig, int w, int h)
{
    data d = {0};
    d.shallow = 0;
    d.w = w;
    d.h = h;
    int i;
    d.X.rows = orig.X.rows;
    d.X.cols = w*h*3;
    d.X.vals = calloc(d.X.rows, sizeof(float*));

    d.y = copy_matrix(orig.y);
#pragma omp parallel for
    for(i = 0; i < orig.X.rows; ++i){
        image im = float_to_image(orig.w, orig.h, 3, orig.X.vals[i]);
        d.X.vals[i] = resize_image(im, w, h).data;
    }
    return d;
}

data load_data_augment(char **paths, int n, int m, char **labels, int k, tree *hierarchy, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.w=size;
    d.h=size;
    d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure, center);
    d.y = load_labels_paths(paths, n, labels, k, hierarchy);
    if(m) free(paths);
    return d;
}

data load_data_tag(char **paths, int n, int m, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.w = size;
    d.h = size;
    d.shallow = 0;
    d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure, 0);
    d.y = load_tags_paths(paths, n, k);
    if(m) free(paths);
    return d;
}

matrix concat_matrix(matrix m1, matrix m2)
{
    int i, count = 0;
    matrix m;
    m.cols = m1.cols;
    m.rows = m1.rows+m2.rows;
    m.vals = calloc(m1.rows + m2.rows, sizeof(float*));
    for(i = 0; i < m1.rows; ++i){
        m.vals[count++] = m1.vals[i];
    }
    for(i = 0; i < m2.rows; ++i){
        m.vals[count++] = m2.vals[i];
    }
    return m;
}

data concat_data(data d1, data d2)
{
    data d = {0};
    d.shallow = 1;
    d.X = concat_matrix(d1.X, d2.X);
    d.y = concat_matrix(d1.y, d2.y);
    d.w = d1.w;
    d.h = d1.h;
    return d;
}

data concat_datas(data *d, int n)
{
    int i;
    data out = {0};
    for(i = 0; i < n; ++i){
        data new = concat_data(d[i], out);
        free_data(out);
        out = new;
    }
    return out;
}

data load_categorical_data_csv(char *filename, int target, int k)
{
    data d = {0};
    d.shallow = 0;
    matrix X = csv_to_matrix(filename);
    float *truth_1d = pop_column(&X, target);
    float **truth = one_hot_encode(truth_1d, X.rows, k);
    matrix y;
    y.rows = X.rows;
    y.cols = k;
    y.vals = truth;
    d.X = X;
    d.y = y;
    free(truth_1d);
    return d;
}

data load_cifar10_data(char *filename)
{
    data d = {0};
    d.shallow = 0;
    long i,j;
    matrix X = make_matrix(10000, 3072);
    matrix y = make_matrix(10000, 10);
    d.X = X;
    d.y = y;

    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);
    for(i = 0; i < 10000; ++i){
        unsigned char bytes[3073];
        fread(bytes, 1, 3073, fp);
        int class = bytes[0];
        y.vals[i][class] = 1;
        for(j = 0; j < X.cols; ++j){
            X.vals[i][j] = (double)bytes[j+1];
        }
    }
    scale_data_rows(d, 1./255);
    //normalize_data_rows(d);
    fclose(fp);
    return d;
}

void get_random_batch(data d, int n, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = rand()%d.X.rows;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}

void get_next_batch(data d, int n, int offset, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = offset + j;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        if(y) memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}

void smooth_data(data d)
{
    int i, j;
    float scale = 1. / d.y.cols;
    float eps = .1;
    for(i = 0; i < d.y.rows; ++i){
        for(j = 0; j < d.y.cols; ++j){
            d.y.vals[i][j] = eps * scale + (1-eps) * d.y.vals[i][j];
        }
    }
}

data load_all_cifar10()
{
    data d = {0};
    d.shallow = 0;
    int i,j,b;
    matrix X = make_matrix(50000, 3072);
    matrix y = make_matrix(50000, 10);
    d.X = X;
    d.y = y;


    for(b = 0; b < 5; ++b){
        char buff[256];
        sprintf(buff, "data/cifar/cifar-10-batches-bin/data_batch_%d.bin", b+1);
        FILE *fp = fopen(buff, "rb");
        if(!fp) file_error(buff);
        for(i = 0; i < 10000; ++i){
            unsigned char bytes[3073];
            fread(bytes, 1, 3073, fp);
            int class = bytes[0];
            y.vals[i+b*10000][class] = 1;
            for(j = 0; j < X.cols; ++j){
                X.vals[i+b*10000][j] = (double)bytes[j+1];
            }
        }
        fclose(fp);
    }
    //normalize_data_rows(d);
    scale_data_rows(d, 1./255);
    smooth_data(d);
    return d;
}

data load_go(char *filename)
{
    FILE *fp = fopen(filename, "rb");
    matrix X = make_matrix(3363059, 361);
    matrix y = make_matrix(3363059, 361);
    int row, col;

    if(!fp) file_error(filename);
    char *label;
    int count = 0;
    while((label = fgetl(fp))){
        int i;
        if(count == X.rows){
            X = resize_matrix(X, count*2);
            y = resize_matrix(y, count*2);
        }
        sscanf(label, "%d %d", &row, &col);
        char *board = fgetl(fp);

        int index = row*19 + col;
        y.vals[count][index] = 1;

        for(i = 0; i < 19*19; ++i){
            float val = 0;
            if(board[i] == '1') val = 1;
            else if(board[i] == '2') val = -1;
            X.vals[count][i] = val;
        }
        ++count;
        free(label);
        free(board);
    }
    X = resize_matrix(X, count);
    y = resize_matrix(y, count);

    data d = {0};
    d.shallow = 0;
    d.X = X;
    d.y = y;


    fclose(fp);

    return d;
}


void randomize_data(data d)
{
    int i;
    for(i = d.X.rows-1; i > 0; --i){
        int index = rand()%i;
        float *swap = d.X.vals[index];
        d.X.vals[index] = d.X.vals[i];
        d.X.vals[i] = swap;

        swap = d.y.vals[index];
        d.y.vals[index] = d.y.vals[i];
        d.y.vals[i] = swap;
    }
}

void scale_data_rows(data d, float s)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        scale_array(d.X.vals[i], d.X.cols, s);
    }
}

void translate_data_rows(data d, float s)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        translate_array(d.X.vals[i], d.X.cols, s);
    }
}

data copy_data(data d)
{
    data c = {0};
    c.w = d.w;
    c.h = d.h;
    c.shallow = 0;
    c.num_boxes = d.num_boxes;
    c.boxes = d.boxes;
    c.X = copy_matrix(d.X);
    c.y = copy_matrix(d.y);
    return c;
}

void normalize_data_rows(data d)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        normalize_array(d.X.vals[i], d.X.cols);
    }
}

data get_data_part(data d, int part, int total)
{
    data p = {0};
    p.shallow = 1;
    p.X.rows = d.X.rows * (part + 1) / total - d.X.rows * part / total;
    p.y.rows = d.y.rows * (part + 1) / total - d.y.rows * part / total;
    p.X.cols = d.X.cols;
    p.y.cols = d.y.cols;
    p.X.vals = d.X.vals + d.X.rows * part / total;
    p.y.vals = d.y.vals + d.y.rows * part / total;
    return p;
}

data get_random_data(data d, int num)
{
    data r = {0};
    r.shallow = 1;

    r.X.rows = num;
    r.y.rows = num;

    r.X.cols = d.X.cols;
    r.y.cols = d.y.cols;

    r.X.vals = calloc(num, sizeof(float *));
    r.y.vals = calloc(num, sizeof(float *));

    int i;
    for(i = 0; i < num; ++i){
        int index = rand()%d.X.rows;
        r.X.vals[i] = d.X.vals[index];
        r.y.vals[i] = d.y.vals[index];
    }
    return r;
}

data *split_data(data d, int part, int total)
{
    data *split = calloc(2, sizeof(data));
    int i;
    int start = part*d.X.rows/total;
    int end = (part+1)*d.X.rows/total;
    data train;
    data test;
    train.shallow = test.shallow = 1;

    test.X.rows = test.y.rows = end-start;
    train.X.rows = train.y.rows = d.X.rows - (end-start);
    train.X.cols = test.X.cols = d.X.cols;
    train.y.cols = test.y.cols = d.y.cols;

    train.X.vals = calloc(train.X.rows, sizeof(float*));
    test.X.vals = calloc(test.X.rows, sizeof(float*));
    train.y.vals = calloc(train.y.rows, sizeof(float*));
    test.y.vals = calloc(test.y.rows, sizeof(float*));

    for(i = 0; i < start; ++i){
        train.X.vals[i] = d.X.vals[i];
        train.y.vals[i] = d.y.vals[i];
    }
    for(i = start; i < end; ++i){
        test.X.vals[i-start] = d.X.vals[i];
        test.y.vals[i-start] = d.y.vals[i];
    }
    for(i = end; i < d.X.rows; ++i){
        train.X.vals[i-(end-start)] = d.X.vals[i];
        train.y.vals[i-(end-start)] = d.y.vals[i];
    }
    split[0] = train;
    split[1] = test;
    return split;
}

