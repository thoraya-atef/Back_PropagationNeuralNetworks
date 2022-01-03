import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.Vector;

public class NN {
    static int M;// m is the number of nodes
    static int L; // l is number of hidden nodes
    static int N;// outputs nodes
    static int K; // number of training examples
    static double X[];
    static double Y[];

    public static Vector<double[]> ReadFile(String nameFile,Vector<double[]> vector) throws FileNotFoundException {
        double mean;
        double StandardDiv;
        double[] Normalized = new double[M];
        File file = new File(nameFile);
        Scanner read = new Scanner(file);
        M = read.nextInt();
        L = read.nextInt();
        N = read.nextInt();
        K = read.nextInt();
        X = new double[M];
        Y = new double[N];
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < M; j++) {
                X[j] = read.nextDouble();
            }
            for (int j = 0; j < N; j++) {
                Y[j] = read.nextDouble();
            }

        }
        vector = GaussianNormalization(X, M);
        return vector;

    }

    // ***************Normalization****************
    public static double CalculateMean(double[] X,int idx, int M) {
        double sum = 0.0;
        for (int i = 0; i < X.length; i++) {
            sum += X[idx];
            idx+=M;
        }
        double mean=sum/M;
        return mean;
    }

    public static double CalculateStandardDiv(double[] X,int idx, int M,double Mean) {
        double standardDiv = 0.0;
        for (int j = idx; j < X.length;) {
            standardDiv+= Math.pow(X[j] - Mean, 2);
            j+=M;
        }
        standardDiv = Math.sqrt(standardDiv / 4);
        return standardDiv;
    }

    public static Vector<double[]> GaussianNormalization(double[] X, int M) {
        Vector<double[]> vector = new Vector<double[]>();
        double normalizedInputs[] = new double[M];
        double Mean;
        double std_dev;
        for (int j = 0; j < K; j++) {
            for (int i = 0; i < M; i++) {
                Mean = CalculateMean(X, i, M);
                std_dev = CalculateStandardDiv(X, i, M,Mean);
                normalizedInputs[i] = (X[(j*M)+i] - Mean) / std_dev;
            }
            vector.add(j, normalizedInputs);
        }
        return vector;
    }
}
