#include "stdafx.h"
#include"BPAnn.h"
#include<stdio.h>
BPAnn::BPAnn(int innum,int ounum,int hdnum){
	in_num = innum;
	ou_num = ounum;
	hd_num = hdnum;
}
double BPAnn::e_pow(double x) {
	double y = 0;
	double xn = 1;
	int nj = 1;
	for (int i = 0; i < 10; i++) {
		y += xn / nj;
		xn *= x;
		nj *= (i + 1);
	}
	return y;
}
double BPAnn::sigmoid(double x) {
	return A / (1 + e_pow(-x/B));

}
void BPAnn::setData(const double input[][NUM], const double label[][NUM], int trainnum) {
	train_num = trainnum;
	for (int i = 0; i < train_num; i++) {
		for (int j = 0; j < in_num; j++)
			this->input[i][j] = input[i][j];
	}
	for (int i = 0; i < train_num; i++) {
		for (int j = 0; j < ou_num; j++)
			this->label[i][j] = label[i][j];
	}
}
void BPAnn::forwardTransfer() {
	//计算隐含层各个节点的输出值  
	for (int j = 0; j < hd_num; j++)
	{
		double t = 0;
		for (int i = 0; i < in_num; i++)
			t += w[1][i][j] * x[0][i];
		t += b[1][j];
		x[1][j] = sigmoid(t);
	}

	//计算输出层各节点的输出值  
	for (int j = 0; j < ou_num; j++)
	{
		double t = 0;
		for (int i = 0; i < hd_num; i++)
			t += w[2][i][j] * x[1][i];
		t += b[2][j];
		x[2][j] = sigmoid(t);
	}

}
double BPAnn::getLoss(int cnt) {
	double ans = 0;
	for (int i = 0; i < ou_num; i++) {
		ans += 0.5*(label[cnt][i] - x[2][i])*(label[cnt][i] - x[2][i]);
	}
	return ans;
}
//计算调整量  
void BPAnn::calcDelta(int cnt)
{
	//计算输出层的delta值  
	for (int i = 0; i < ou_num; i++)
		d[2][i] = (x[2][i] - label[cnt][i]) * x[2][i] * (A - x[2][i]) / (A * B);
	//计算隐含层的delta值  
	for (int i = 0; i < hd_num; i++)
	{
		double t = 0;
		for (int j = 0; j < ou_num; j++)
			t += w[2][i][j] * d[2][j];
		d[1][i] = t * x[1][i] * (A - x[1][i]) / (A * B);
	}
}

//根据计算出的调整量对BP网络进行调整  
void BPAnn::updateNetWork()
{
	//隐含层和输出层之间权值和阀值调整  
	for (int i = 0; i < hd_num; i++)
	{
		for (int j = 0; j < ou_num; j++)
			w[2][i][j] -= ETA_W * d[2][j] * x[1][i];
	}
	for (int i = 0; i < ou_num; i++)
		b[2][i] -= ETA_B * d[2][i];

	//输入层和隐含层之间权值和阀值调整  
	for (int i = 0; i < in_num; i++)
	{
		for (int j = 0; j < hd_num; j++)
			w[1][i][j] -= ETA_W * d[1][j] * x[0][i];
	}
	for (int i = 0; i < hd_num; i++)
		b[1][i] -= ETA_B * d[1][i];
}
//误差信号反向传递子过程  
void BPAnn::reverseTransfer(int cnt)
{
	calcDelta(cnt);
	updateNetWork();
}

//计算所有样本的精度  
double BPAnn::getLoss()
{
	double ans = 0;
	
	for (int i = 0; i < train_num; i++)
	{
		
		for (int j = 0; j <in_num; j++)
			x[0][j] = input[i][j];
		forwardTransfer();
		
		for (int j = 0; j < ou_num; j++)
			ans += 0.5 * (x[2][j] - label[i][j]) * (x[2][j] - label[i][j]);
	}
	return ans / train_num;
}
//开始进行训练  
void BPAnn::train()
{
	printf("Begin to train BP NetWork!\n");
	
	initNetWork();


	for (int iter = 0; iter <= ITERS; iter++)
	{
		for (int cnt = 0; cnt < train_num; cnt++)
		{
			//第一层输入节点赋值  
			for (int i = 0; i < in_num; i++)
				x[0][i] = input[cnt][i];

			while (1)
			{
				forwardTransfer();
				if (getLoss(cnt) < LOSS)    //如果误差比较小，则针对单个样本跳出循环  
					break;
				reverseTransfer(cnt);
			}
		}
		printf("This is the %d th trainning NetWork !\n", iter);

		double loss = getLoss();
		printf("All Samples loss is %lf\n", loss);
		if (loss < LOSSU) break;
	}
	printf("The BP NetWork train End!\n");
}

//根据训练好的网络来预测输出值  
void BPAnn::foreCast(const double input[],double y[])
{
	
	for (int i = 0; i < in_num; i++)
		x[0][i] = input[i];

	forwardTransfer();

	for (int i = 0; i < ou_num; i++)
		y[i] = x[2][i];
}
void BPAnn::initNetWork() {
	for (int i = 0; i < LAYER; i++) {
		for (int j = 0; j < NUM; j++)
		{
			for (int k = 0; k < NUM; k++) {
				w[i][j][k] = 0.1;
			}
			b[i][j] = 0.0;
		}
	}

}