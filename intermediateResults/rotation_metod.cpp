#include <array>
#include <numbers>
#include <source_location>
#include <string>
#include <vector>
#include <iomanip>


#include "../include/Generators/RealNumberGenerator.hpp"
#include "../include/Particles/Particles.hpp"

using std::cout;

void rotate_my(MathVector &a_vector, double beta){
  MathVector rotation_y[3];
  cout<<"beta"<<beta<<"\n";
  rotation_y[0] = MathVector(cos(beta),0.,sin(beta));
  rotation_y[1] = MathVector(0.,1.,0.);
  rotation_y[2] = MathVector(-sin(beta),0.,cos(beta));
  
  double components[3];
  for(int i{0}; i<3 ; i++){
    cout<<"my components"<<rotation_y[i]<<"\n";
    components[i]=rotation_y[i]*a_vector;
  }

  a_vector.setX(components[0]);
  a_vector.setY(components[1]);
  a_vector.setZ(components[2]);
  cout << std::setprecision(30)<<"rotated vector:\t"<<a_vector<<"\n";
}

void rotate_mz(MathVector &a_vector, double gamma){
  MathVector rotation_z[3];                               
  rotation_z[0] = MathVector(cos(gamma),-sin(gamma), 0.); 
  rotation_z[1] = MathVector(sin(gamma),cos(gamma), 0.);  
  rotation_z[2] = MathVector(0.       ,0.       , 1.);    
  
  double components[3];
  for(int i{0}; i<3 ; i++){
    components[i]=rotation_z[i]*a_vector;
  }

  a_vector.setX(components[0]);
  a_vector.setY(components[1]);
  a_vector.setZ(components[2]);
  cout<<std::setprecision(30)<<"rotated vector:\t"<<a_vector<<"\n";
}

void rotation(MathVector &a_vector, double &beta, double &gamma , std::string str){

 cout<<str<<" option \n";
  
 if(str == "align"){
    beta  =acos(a_vector.getZ() / a_vector.module());
    gamma =atan2(a_vector.getY(), a_vector.getX());
    beta*=-1;
    gamma*=-1;
    rotate_mz(a_vector, gamma);
    rotate_my(a_vector,beta);
    beta*=-1;
    gamma*=-1;
  }
  else{
    rotate_my(a_vector,beta);
    rotate_mz(a_vector, gamma);
  }

}


int main(){
  
  MathVector a_vector(1,1,1);
  cout<<a_vector<<"\ncontinue\n";
  double beta{}, gamma{};
  rotation(a_vector, beta, gamma, "align");
  rotation(a_vector, beta, gamma, "");

  return 0;
}
