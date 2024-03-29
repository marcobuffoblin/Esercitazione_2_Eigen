#include <iostream>
#include "Eigen/Eigen"
#include<iomanip>
using namespace std;
using namespace Eigen;

double err_PALU(Matrix2d &A , Vector2d &b, Vector2d &sol){
    PartialPivLU<Matrix2d> lu(A);
    Matrix2d L = Matrix2d::Identity();
    L.triangularView<StrictlyLower>() = lu.matrixLU();
    Matrix2d U = lu.matrixLU().triangularView<Upper>();
    Matrix2d P=lu.permutationP();
    //cout << "La matrice L proveniente dalla fattorizzazione e':"<<endl << L << endl;
    //cout << "La matrice U proveniente dalla fattorizzazione e':"<<endl << U << endl;
    //cout << "La matrice P proveniente dalla fattorizzazione e':"<<endl << P << endl;
    VectorXd y =L.lu().solve(P*b);
    VectorXd x =U.lu().solve(y);
    //cout<<"La soluzione del sistema utilizzando la PA=LU : "<<endl<<setprecision(16)<<xLU<<endl;
    double err_LU= ((x-sol).norm())/(sol.norm());
    return err_LU;
}

double err_QR(Matrix2d &A , Vector2d &b, Vector2d &sol){
    HouseholderQR<Matrix2d> qr(A);
    Matrix2d Q= qr.householderQ();
    Matrix2d R= qr.matrixQR().triangularView<Upper>();
    //cout << "La matrice Q proveniente dalla fattorizzazione e':"<<endl << Q << endl;
    //cout << "La matrice R proveniente dalla fattorizzazione e':"<<endl << R << endl;
    VectorXd x =R.householderQr().solve(Q.transpose()*b);
    //cout<<"La soluzione del sistema utilizzando la QR : "<<endl<<setprecision(16)<<xQR<<endl;
    double err_QR= ((x-sol).norm())/(sol.norm());
    return err_QR;
}



int main()
{
    // Creo la matrice A, il vettore b e la soluzione sol
    Matrix2d A1{{5.547001962252291e-01, -3.770900990025203e-02},{ 8.320502943378437e-01, -9.992887623566787e-01}};
    Vector2d b1(-5.169911863249772e-01, 1.672384680188350e-01);
    Matrix2d A2{{5.547001962252291e-01, -5.540607316466765e-01},{8.320502943378437e-01, -8.324762492991313e-01}};
    Vector2d b2(-6.394645785530173e-04, 4.259549612877223e-04);
    Matrix2d A3{{5.547001962252291e-01, -5.547001955851905e-01},{8.320502943378437e-01, -8.320502947645361e-01}};
    Vector2d b3(-6.400391328043042e-10, 4.266924591433963e-10);
    Vector2d sol(-1.0, -1.0);


    // Stampiamo la soluzione
    cout<<"L'errore relativo al sistema numero 1, utilizzando la PA=LU: "<<endl<<setprecision(16)<<scientific<<err_PALU(A1,b1,sol)<<endl;
    cout<<"L'errore relativo al sistema numero 1, utilizzando la QR : "<<endl<<setprecision(16)<<scientific<<err_QR(A1,b1,sol)<<endl<<endl;
    cout<<"L'errore relativo al sistema numero 2, utilizzando la PA=LU: "<<endl<<setprecision(16)<<scientific<<err_PALU(A2,b2,sol)<<endl;
    cout<<"L'errore relativo al sistema numero 2, utilizzando la QR : "<<endl<<setprecision(16)<<scientific<<err_QR(A2,b2,sol)<<endl<<endl;
    cout<<"L'errore relativo al sistema numero 3, utilizzando la PA=LU: "<<endl<<setprecision(16)<<scientific<<err_PALU(A3,b3,sol)<<endl;
    cout<<"L'errore relativo al sistema numero 3, utilizzando la QR : "<<endl<<setprecision(16)<<scientific<<err_QR(A3,b3,sol)<<endl;

    return 0;
}
