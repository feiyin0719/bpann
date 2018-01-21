#pragma once
#define LAYER    3        //三层神经网络  
#define NUM      10       //每层的最多节点数  
#define A        30.0  
#define B        10.0     //A和B是S型函数的参数  
#define ITERS    1000     //最大训练次数  
#define ETA_W    0.0035   //权值调整率  
#define ETA_B    0.001    //阀值调整率  
#define LOSS    0.002    //单个样本允许的误差  
#define LOSSU     0.005    //每次迭代允许的误差
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
		int in_num;                 //输入层节点数  
		int ou_num;                 //输出层节点数  
		int hd_num;                 //隐含层节点数  
		int train_num;
		double w[LAYER][NUM][NUM];
		double b[LAYER][NUM];
		double x[LAYER][NUM];
		double d[LAYER][NUM];
		double label[MAX_TRAIN_NUM][NUM];
		double input[MAX_TRAIN_NUM][NUM];
		void initNetWork();
		void forwardTransfer();     //正向传播子过程  
		void reverseTransfer(int);  //逆向传播子过程  
		void calcDelta(int);        //计算w和b的调整量  
		void updateNetWork();       //更新权值和阀值  
		double getLoss(int);         //计算单个样本的误差  
		double getLoss();             //计算所有样本的精度  

};