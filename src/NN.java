import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.Vector;

public class NN {
    static int M;// m is the number of nodes
    static int L; // l is number of hidden nodes
    static int N;// outputs nodes
    static int K; // number of training examples
    static Vector<Double> X;
    static Vector<Double> Y;

    public static  Vector<double[]> ReadFile(String nameFile, Vector<double[]> vector) throws FileNotFoundException {
        File file = new File(nameFile);
        Scanner read = new Scanner(file);
        M = read.nextInt();
        L = read.nextInt();
        N = read.nextInt();
        K = read.nextInt();
        X = new Vector<Double>();
        Y = new Vector<Double>();
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < M; j++) {
                X.add(read.nextDouble());
            }
            for (int j = 0; j < N; j++) {
                Y.add(read.nextDouble());
            }

        }
        vector = GaussianNormalization(X, M);
        return vector;

    }

    // ***************Normalization****************
    public static double CalculateMean(Vector<Double> X, int idx) {
        double sum = 0.0;
        for (int i = 0; i < K;i++) {
            sum += X.get((i * M) + idx);
        }
        double mean=sum/K;
        return mean;
    }

    public static double CalculateStandardDiv(Vector<Double> X, int idx, double Mean) {
        double standardDiv = 0.0;
        for (int j = 0; j < K;j++) {
            standardDiv+= Math.pow(X.get((j * M) + idx) - Mean, 2);
        }
        standardDiv = Math.sqrt(standardDiv / K);
        return standardDiv;
    }

    public static Vector<double[]> GaussianNormalization(Vector<Double> X, int M) {
        Vector<double[]> vector = new Vector<>();
        double normalizedInputs[] = new double[M];
        double[]  Mean = new double[M];
        double[] std_dev = new double[M];
        for (int i = 0; i < M; i++){
            Mean[i] = CalculateMean(X, i);
            std_dev[i] = CalculateStandardDiv(X, i,Mean[i]);
        }
        for (int j = 0; j < K; j++) {
            for (int i = 0; i < M; i++) {
                normalizedInputs[i] = (X.get((j * M) + i) -  Mean[i]) / std_dev[i];
            }
            vector.add(j, normalizedInputs);
        }
        return vector;
    }
}
