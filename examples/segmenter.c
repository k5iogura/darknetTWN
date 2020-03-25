#include "darknet.h"
#include <sys/time.h>
#include <assert.h>

void train_segmenter(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int display)
{
    int i;

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];
    image pred = get_network_image(net);

    int div = net->w/pred.w;
    assert(pred.w * div == net->w);
    assert(pred.h * div == net->h);

    int imgs = net->batch * net->subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *train_list = option_find_str(options, "train", "data/train.list");

    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    //args.threads = 32;
    args.threads = 1;
    args.scale = div;

    args.min = net->min_crop;
    args.max = net->max_crop;
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;
    //args.classes = 80;
    args.classes = 1;

    args.paths = paths;
    args.n = imgs;
    args.m = N;
    args.type = SEGMENTATION_DATA;

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    int epoch = (*net->seen)/N;
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        double time = what_time_is_it_now();

        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        //printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
        time = what_time_is_it_now();

        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if(display){
            image tr = float_to_image(net->w/div, net->h/div, 80, train.y.vals[net->batch*(net->subdivisions-1)]);
            image im = float_to_image(net->w, net->h, net->c, train.X.vals[net->batch*(net->subdivisions-1)]);
            image mask = mask_to_rgb(tr);
            image prmask = mask_to_rgb(pred);
            show_image(im, "input", 1);
            show_image(prmask, "pred", 1);
            show_image(mask, "truth", 100);
            free_image(mask);
            free_image(prmask);
        }
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, *net->seen);
        free_data(train);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
        }
        if(get_current_batch(net)%100 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);

    free_network(net);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

float map_cpu(float thresh, image iG, float* iP){
    static int i, c, FP=0, TP=0, FN=0;
    for(c=0;c<iG.c;c++)
        for(i=iG.w*iG.h*c;i<iG.w*iG.h;i++){
            float prv = (iP[i]>thresh)? 1.0 : 0.0;
            float dff = iG.data[i] - prv*(c+1);
            float adf = fabs(dff);
            if(adf<1.0e-6) TP++;
            else if(dff<0) FP++;
            else if(dff>0) FN++;
        }
    int numerator=TP, denominator=TP+FN+FP;
    if(!denominator) return 0.0;
    return 100.*numerator/(float)denominator;
}

void map_segmenter(char *datafile, char *cfg, char *weights, float thresh, int display)
{
    list *options = read_data_cfg(datafile);
    char *valid_list = option_find_str(options, "valid", "data/valid.list");
    list *plist = get_paths(valid_list);
    char **paths = (char **)list_to_array(plist);
    printf("validate %d images\n", plist->size);
    int N = plist->size;
    int i;

    network *net = load_network(cfg, weights, 0);
    set_batch_network(net, 1);

    clock_t time;
    char buff[256];
    char *input = buff;
    float total_time=0;
    float map=0;
    printf("%f threshold\n0000000",thresh);
    for(i=0;i<N;i++){
        strncpy(input, paths[i], 256);
        find_replace(paths[i], "SegmentationClass", "JPEGImages", input);
        find_replace(input, ".png", ".jpg", input);

        image im = load_image_color(input, 0, 0);
        image sized = letterbox_image(im, net->w, net->h);

        image gtmask = load_image(paths[i], sized.w, sized.h, 1);

        float *X = sized.data;
        time=clock();
        float *predictions = network_predict(net, X);
        image pred = get_network_image(net);
        image prmask = mask_to_rgb(pred);
        assert(gtmask.h==pred.h && gtmask.w==pred.w && gtmask.c==pred.c);
        total_time+=sec(clock()-time);
        map=map_cpu(thresh, gtmask,predictions);
        printf("\b\b\b\b\b\b\b%4.1fmAP",map);
        fflush(stdout);
        if(display){
            printf("infer %s\n",input);
            printf("gtrth %s\n",paths[i]);
            show_image(sized, "orig", 1);
            show_image(prmask, "pred", 0);
        }
        free_image(im);
        free_image(sized);
        free_image(prmask);
        free_image(gtmask);
    }
    printf("\n%9.2f mAP Predicted in %6.2f seconds %9.2f FPS.\n",map ,total_time, N/total_time);
}

void predict_segmenter(char *datafile, char *cfg, char *weights, char *filename)
{
    network *net = load_network(cfg, weights, 0);
    set_batch_network(net, 1);
    srand(2222222);

    clock_t time;
    char buff[256];
    char *input = buff;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input, 0, 0);
        image sized = letterbox_image(im, net->w, net->h);

        float *X = sized.data;
        time=clock();
        float *predictions = network_predict(net, X);
        image pred = get_network_image(net);
        image prmask = mask_to_rgb(pred);
        printf("Predicted: %f\n", predictions[0]);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        show_image(sized, "orig", 1);
        show_image(prmask, "pred", 0);
        free_image(im);
        free_image(sized);
        free_image(prmask);
        if (filename) break;
    }
}

void demo_segmenter(char *datacfg, char *cfg, char *weights, int cam_index, const char *filename)
{
#ifdef OPENCV
    printf("Classifier Demo\n");
    network *net = load_network(cfg, weights, 0);
    set_batch_network(net, 1);

    srand(2222222);
    void * cap = open_video_stream(filename, cam_index, 0,0,0);

    if(!cap) error("Couldn't connect to webcam.\n");
    float fps = 0;

    while(1){
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);

        image in = get_image_from_stream(cap);
        image in_s = letterbox_image(in, net->w, net->h);

        network_predict(net, in_s.data);

        printf("\033[2J");
        printf("\033[1;1H");
        printf("\nFPS:%.0f\n",fps);

        image pred = get_network_image(net);
        image prmask = mask_to_rgb(pred);
        show_image(prmask, "Segmenter", 10);
        
        free_image(in_s);
        free_image(in);
        free_image(prmask);

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
#endif
}


void run_segmenter(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int clear = find_arg(argc, argv, "-clear");
    int display = find_arg(argc, argv, "-display");
    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    float thresh = find_float_arg(argc,argv,"-thresh",0.01);
    if(0==strcmp(argv[2], "test")) predict_segmenter(data, cfg, weights, filename);
    else if(0==strcmp(argv[2], "train")) train_segmenter(data, cfg, weights, gpus, ngpus, clear, display);
    else if(0==strcmp(argv[2], "demo")) demo_segmenter(data, cfg, weights, cam_index, filename);
    else if(0==strcmp(argv[2], "map")) map_segmenter(data, cfg, weights, thresh, display);
}


