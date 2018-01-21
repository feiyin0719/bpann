#pragma once
#define LAYER    3        //����������  
#define NUM      10       //ÿ������ڵ���  
#define A        30.0  
#define B        10.0     //A��B��S�ͺ����Ĳ���  
#define ITERS    1000     //���ѵ������  
#define ETA_W    0.0035   //Ȩֵ������  
#define ETA_B    0.001    //��ֵ������  
#define LOSS    0.002    //����������������  
#define LOSSU     0.005    //ÿ�ε�����������
#define MAX_TRAIN_NUM 1000
class BPAnn {
	public:
		double e_pow(double x);
		double sigmoid(double x);
		void setData(const double input[][NUM] ,const double label[][NUM],int trainnum);
		void train();
		void foreCast(const double x[],double y[]);
		BPAnn(int innum,int ounum,int hdnum);
	private:
		int in_num;                 //�����ڵ���  
		int ou_num;                 //�����ڵ���  
		int hd_num;                 //������ڵ���  
		int train_num;
		double w[LAYER][NUM][NUM];
		double b[LAYER][NUM];
		double x[LAYER][NUM];
		double d[LAYER][NUM];
		double label[MAX_TRAIN_NUM][NUM];
		double input[MAX_TRAIN_NUM][NUM];
		void initNetWork();
		void forwardTransfer();     //���򴫲��ӹ���  
		void reverseTransfer(int);  //���򴫲��ӹ���  
		void calcDelta(int);        //����w��b�ĵ�����  
		void updateNetWork();       //����Ȩֵ�ͷ�ֵ  
		double getLoss(int);         //���㵥�����������  
		double getLoss();             //�������������ľ���  

};