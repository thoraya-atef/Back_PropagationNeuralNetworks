import java.io.*;
import java.util.Random;
import java.util.Scanner;
import java.util.Vector;

public class NN {
    static int M,L,N,K;// m is the number of inputs nodes// l is number of hidden nodes// outputs nodes
    //K number of training examples
    static Vector<double[]> X;
    static Vector<double[]> Y;
    static Vector<double[][]> Weights_hidden_layer;
    static Vector<double[][]> Weights_output_layer;

    public static  Vector<double[]> ReadFile(String nameFile, Vector<double[]> vector) throws FileNotFoundException {
        File file = new File( nameFile+".txt");
        Scanner read = new Scanner(file);
        M = read.nextInt();
        L = read.nextInt();
        N = read.nextInt();
        K = read.nextInt();
        X = new Vector<double[]>();
        Y = new Vector<double[]>();
        double[] x;
        double[] y;
        for (int i = 0; i < K; i++) {
            x=new double[M];
            y=new double[N];
            for (int j = 0; j < M; j++) {
                x[j]=read.nextDouble();
            }
            X.add(i,x);
            for (int j = 0; j < N; j++) {
                y[j]=read.nextDouble();
            }
            Y.add(i,y);

        }
        vector = GaussianNormalization(X, M);

        return vector;

    }

    // ***************Normalization****************
    public static double CalculateMean(Vector<double[]> X, int idx) {
        double sum = 0.0;
        for (int i = 0; i < K;i++) {
            sum += X.get(i)[idx];
        }
        double mean=sum/K;
        return mean;
    }

    public static double CalculateStandardDiv(Vector<double[]> X, int idx, double Mean) {
        double standardDiv = 0.0;
        for (int j = 0; j < K;j++) {
            standardDiv+= Math.pow(X.get(j)[idx] - Mean, 2);
        }
        standardDiv = Math.sqrt(standardDiv / K);
        return standardDiv;
    }

    public static Vector<double[]> GaussianNormalization(Vector<double[]> X, int M) {
        Vector<double[]> vector = new Vector<>();
        double normalized[] = new double[M];
        double[]  Mean = new double[M];
        double[] std_dev = new double[M];
        for (int i = 0; i < M; i++){
            Mean[i] = CalculateMean(X, i);
            std_dev[i] = CalculateStandardDiv(X, i,Mean[i]);
        }
        for (int j = 0; j < K; j++) {
            for (int i = 0; i < M; i++) {
                normalized[i] = (X.get(j)[i] -  Mean[i]) / std_dev[i];
            }
            vector.add(j, normalized);
        }
        return vector;
    }
    ///********************Wights***************//
    public static void writeWeights() {
        File file = new File("Weights.txt");
        try (PrintWriter weights = new PrintWriter(file)) {
            weights.print("Weights Of Hidden Layer  : "+"\n");
            weights.print("---------------------------------"+"\n");
            for(int i=0; i<K; i++) {
                for (int j = 0; j < L; j++) {
                    for (int z = 0; z < M; z++) {
                        weights.print(Weights_hidden_layer.get(i)[j][z] + "  ");
                    }
                }
                weights.print("\n");
            }
             weights.print("---------------------------------"+"\n");
            weights.print("Weights Of Output Layer  : "+"\n");
            weights.print("---------------------------------"+"\n");
            for(int i=0; i<K; i++){
                for (int j = 0; j < N; j++) {
                    for (int z = 0; z <L; z++) {
                        weights.print(Weights_output_layer.get(i)[j][z] + "  ");
                    }
                }
                weights.print("\n");
            }
        }catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
    public static double[][] Weights(double weights[][], int n, int m){
        Random rand_num = new Random();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                //get random number of weights from -2 to 2
                //by using this rule:rangeMin + (rangeMax - rangeMin) * rand_num.nextDouble()
                weights[i][j] = (-2) + (4) * rand_num.nextDouble();
            }
        }
        return weights;
    }
    public static double[] SumofXdotW(double[][] weights,double[] inputs){
        double[] result = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                result[i] += weights[i][j] * inputs[j];
            }
        }
        return result;
    }
    public static double SigmoidFunction(double var){
        double sigmoid =(1.0 / (1 + Math.exp(-var)));
        return sigmoid;
    }
    public static double[] FeedForward_hidden(double weights[][], double inputs[]){
        double result[] = new double[weights.length];
        double[] resultofsum = SumofXdotW(weights, inputs);
        for (int i = 0; i < L; i++) {
            result[i] = SigmoidFunction(resultofsum[i]);
        }
        return result;
    }
    public static double[] FeedForward_output(double weights[][], double Hidden[]){
        double result[] = new double[weights.length];
        double[] resultofsum = SumofXdotW(weights, Hidden);
        for (int i = 0; i < resultofsum.length; i++) {
            result[i] = SigmoidFunction(resultofsum[i]);
        }
        return result;
    }
    public static double[] ErrorofOutputlayer(double Y_target[], double predicted[]){
        double Error[] = new double[N];
        double Y_error[] = new double[N];
        for (int i = 0; i < N; i++) {
            Error[i] = predicted[i] - Y_target[i];
        }
        for (int i = 0; i < N; i++) {
            Y_error[i] = predicted[i] * (1 - predicted[i]) * Error[i];
        }
        return Y_error;
    }
    public static double[] ErrorofHiddenlayer(double Y_error[],double H[], double Weights_hiddenLayer[][]){
        double Hidden_error[] = new double[L];
        for (int i = 0; i < L; i++) {
            double Sum = 0;
            for (int j = 0; j < N; j++) {
                Sum += (Y_error[j] * Weights_hiddenLayer[i][j]);
            }
            Hidden_error[i] = H[i] * (1 - H[i]) * Sum;
        }
        return Hidden_error;
    }
    public static double[][] Updated_weights_of_outputlayer(double[][] old_Weights, double learning_rate,
                                                     double[] L_out, double[] Y_error){

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < L; j++) {
                old_Weights[i][j] = old_Weights[i][j] - learning_rate * L_out[j] * Y_error[i];
            }
        }
        return old_Weights;
    }
    public static double[][] Updated_weights_of_Hiddenlayer(double[][] old_Weights, double learning_rate,
                                                     double[] M_out, double[] Hidden_error){
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < M; j++) {
                old_Weights[i][j] = old_Weights[i][j] - learning_rate * M_out[j] * Hidden_error[i];

            }
        }
        return old_Weights;
    }
    public static double Mean_Sq_error(double Y_targetl[], double Y_Predicted[]){
        double error = 0;
        for (int i = 0; i < N; i++) {
            error += Math.pow(Y_targetl[i] - Y_Predicted[i], 2);
        }
        error /= N;
        return error;
    }
    ///*******************First Program*****************
    public static void FirstProgram() throws IOException {
        Vector<double[]>v= new Vector<>();
        int num_of_iterations=500;
        String name="Train";
        v= ReadFile(name,v);
        double Hidden[] = new double[L];
        double Output[] = new double[N];
        double Errors_Hs[] = new double[L];
        double Errors_Ys[] = new double[N];
        double WeightsHidden[][]=new double[L][M];
        double WeightsOutput[][]=new double[N][L];
        Weights_hidden_layer=new Vector<>();
        Weights_output_layer=new Vector<>();
        double learning_rate = 0.1;
        double MSerors[] = new double[K];
        WeightsHidden = Weights(WeightsHidden,L,M);

        WeightsOutput = Weights(WeightsOutput,N,L);
        for (int i=0;i<num_of_iterations;i++){
            for(int j=0;j<K;j++){
                Hidden=FeedForward_hidden(WeightsHidden,v.get(j));
                Output=FeedForward_output(WeightsOutput,Hidden);
                Errors_Ys=ErrorofOutputlayer(Y.get(j),Output);
                Errors_Hs=ErrorofHiddenlayer(Errors_Ys,Hidden,WeightsHidden);
                WeightsOutput=Updated_weights_of_outputlayer(WeightsOutput,learning_rate,Hidden,Errors_Ys);
                WeightsHidden=Updated_weights_of_Hiddenlayer(WeightsHidden,learning_rate,v.get(j),Errors_Hs);
                if(i==num_of_iterations-1) {
                    Weights_hidden_layer.add(j, WeightsHidden);
                    Weights_output_layer.add(j, WeightsOutput);
                    MSerors[j] = Mean_Sq_error(Y.get(j), Output);
                    System.out.println("Mean Square Error of  = " +(j+1)+ MSerors[j] + "\n");
                }

            }

        }
    }
    public static void SecondProgram() throws IOException {
        Vector<double[]>v= new Vector<>();
        String name="Train";
        v= ReadFile(name,v);
        double Hidden[] = new double[L];
        double Output[] = new double[N];
        double MSerors[] = new double[K];
        System.out.println("-----------------------------------------------");
        System.out.println("Mean Square Error in  Second Program \n");
        System.out.println("---------------------------------------------------");
        for(int j=0;j<K;j++) {
            Hidden = FeedForward_hidden(Weights_hidden_layer.get(j), v.get(j));
            Output = FeedForward_output(Weights_output_layer.get(j), Hidden);
            MSerors[j] = Mean_Sq_error(Y.get(j), Output);
            System.out.println("Mean Square Error of  = " +(j+1)+ MSerors[j] + "\n");
        }

    }


    //********************Main*******************//
    public static void main(String[] args) throws IOException {
        NN Neural_N = new NN();
        Neural_N.FirstProgram();
        Neural_N.writeWeights();
        Neural_N.SecondProgram() ;

    }
}
