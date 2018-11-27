#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>


using namespace std;

void comma2space(char *s){
	int i=0;
	while(s[i]!=0){
		if(s[i] == ',')
			s[i] = '&';
		i++;
	}
}


int main(int argc, char * argv[]){
	//abrir arquivo csv
	ifstream in(argv[1]);
	if(!in.is_open()){
		cerr << "Erro ao abrir arquivo: "<<argv[1] << '\n';
		return -1;
	}
	ofstream out(argv[2]);

	if(!out.is_open()){
		cerr << "Erro ao abrir arquivo: "<<argv[2] << '\n';
		return -1;
	}
	
	char *s;
	//stringstream ss;
	//ss.str( "\\begin{bmatrix}\n");
	//out << ss.str();
	while(in.good()){
		in.getline(s,256,'\n');
		//cout << s<<endl;
		comma2space(s);
		//cout << s << endl;
		strcat(s,"\\\\\n");
		
//		out << s;
	}
	try{
		out << "\\end{bmatrix}";
	}
	catch(...){
		cout << "ero\n";
	}
	in.close();
	out.close();
	return 0;
}
