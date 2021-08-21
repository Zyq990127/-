#include <stdbool.h>
#include <inttypes.h>
#include <math.h>
#include <Gap.h>
typedef struct{
	uint32_t x;
	uint32_t y;
	uint32_t w;
	uint32_t h;
	int16_t score;
	uint16_t class;
	uint8_t alive;
}bbox_t;
typedef struct{
	float x;
	float y;
	float w;
	float h;
	int16_t score;
	uint16_t class;
	uint8_t alive;
}bbox_fp_t;
typedef struct{
	bbox_t * bbs;
	int16_t num_bb;
}bboxs_t;
typedef struct{
	bbox_fp_t * bbs;
	int16_t num_bb;
}bboxs_fp_t;
typedef struct {
	float w;
	float h;
}anchorWH_t;


void convertCoordBboxes(bboxs_t *boundbxs,float scale_x,float scale_y);

void non_max_suppress(bboxs_t * boundbxs);

void drawBboxes(bboxs_t *boundbxs, uint8_t *img);

void printBboxes(bboxs_t *boundbxs);

void printBboxes_forPython(bboxs_t *boundbxs);

int rect_intersect_area( short a_x, short a_y, short a_w, short a_h,
                         short b_x, short b_y, short b_w, short b_h );