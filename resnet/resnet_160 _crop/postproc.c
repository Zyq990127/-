#include<stdio.h>
#include"postproc.h"

#define NON_MAX_THRES 10

void convertCoordBboxes(bboxs_t *boundbxs,float scale_x,float scale_y){
    for (int counter=0;counter< boundbxs->num_bb;counter++){
        //Set Alive for non max this should be done somewhere else
        boundbxs->bbs[counter].alive=1;
        boundbxs->bbs[counter].x = (int)(FIX2FP(boundbxs->bbs[counter].x,20) * scale_x);
        boundbxs->bbs[counter].y = (int)(FIX2FP(boundbxs->bbs[counter].y,20) * scale_y);
        boundbxs->bbs[counter].w = (int)(FIX2FP(boundbxs->bbs[counter].w,26) * scale_x);
        boundbxs->bbs[counter].h = (int)(FIX2FP(boundbxs->bbs[counter].h,26) * scale_y);
        boundbxs->bbs[counter].x = boundbxs->bbs[counter].x - (boundbxs->bbs[counter].w/2);
        boundbxs->bbs[counter].y = boundbxs->bbs[counter].y - (boundbxs->bbs[counter].h/2);
    }
}


void non_max_suppress(bboxs_t * boundbxs){

    int idx,idx_int;

    //Non-max supression
     for(idx=0;idx<boundbxs->num_bb;idx++){
        //check if rect has been removed (-1)
        if(boundbxs->bbs[idx].alive==0)
            continue;
 
        for(idx_int=0;idx_int<boundbxs->num_bb;idx_int++){

            if(boundbxs->bbs[idx_int].alive==0 || idx_int==idx)
                continue;

            //check the intersection between rects
            int intersection = rect_intersect_area(boundbxs->bbs[idx].x,boundbxs->bbs[idx].y,boundbxs->bbs[idx].w,boundbxs->bbs[idx].h,
                                                   boundbxs->bbs[idx_int].x,boundbxs->bbs[idx_int].y,boundbxs->bbs[idx_int].w,boundbxs->bbs[idx_int].h);

            if(intersection >= NON_MAX_THRES){ //is non-max
                //supress the one that has lower score that is alway the internal index, since the input is sorted
                boundbxs->bbs[idx_int].alive =0;
            }
        }
    }
}


void drawBboxes(bboxs_t *boundbxs, uint8_t *img){
    for (int counter=0;counter< boundbxs->num_bb;counter++){
        if(boundbxs->bbs[counter].alive){
            DrawRectangle(img, 120, 160, boundbxs->bbs[counter].x, boundbxs->bbs[counter].y, boundbxs->bbs[counter].w, boundbxs->bbs[counter].h, 255);
        }
    }
}


void printBboxes(bboxs_t *boundbxs)
{
    printf("\n\n======================================================");
    printf("\nDetected Bounding boxes                                 ");
    printf("\n======================================================\n");
    printf("BoudingBox:  score     cx     cy     w     h    class");
    printf("\n------------------------------------------------------\n");

    for (int counter=0;counter< boundbxs->num_bb;counter++){
        if(boundbxs->bbs[counter].alive)
            printf("bbox [%02d] : %.5f     %03d    %03d     %03d    %03d     %02d\n",
                counter,
                FIX2FP(boundbxs->bbs[counter].score,15 ),
                boundbxs->bbs[counter].x,
                boundbxs->bbs[counter].y,
                boundbxs->bbs[counter].w,
                boundbxs->bbs[counter].h,
                boundbxs->bbs[counter].class);
    }
}


void printBboxes_forPython(bboxs_t *boundbxs)
{
    printf("\n\n======================================================");
    printf("\nThis can be copy-pasted to python to draw BoudingBoxs   ");
    printf("\n\n");

    for (int counter=0;counter< boundbxs->num_bb;counter++){
        if(boundbxs->bbs[counter].alive)
            printf("rect = patches.Rectangle((%d,%d),%d,%d,linewidth=1,edgecolor='r',facecolor='none')\nax.add_patch(rect)\n",
                boundbxs->bbs[counter].x,
                boundbxs->bbs[counter].y,
                boundbxs->bbs[counter].w,
                boundbxs->bbs[counter].h);
    }
}


int rect_intersect_area( short a_x, short a_y, short a_w, short a_h,
                         short b_x, short b_y, short b_w, short b_h )
{
    #define MIN(a,b) ((a) < (b) ? (a) : (b))
    #define MAX(a,b) ((a) > (b) ? (a) : (b))

    int x = MAX(a_x,b_x);
    int y = MAX(a_y,b_y);

    int size_x = MIN(a_x+a_w,b_x+b_w) - x;
    int size_y = MIN(a_y+a_h,b_y+b_h) - y;

    if(size_x <=0 || size_x <=0)
        return 0;
    else
        return size_x*size_y; 

    #undef MAX
    #undef MIN
}
