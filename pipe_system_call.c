#include<stdio.h>
#include<unistd.h>
#include<stdlib.h>
#define MSGSIZE 16
char* msg1="read-0 ";
char* msg2="write-1";
char* msg3="error-2";
char inbuf[MSGSIZE];
int i;
void Write();
char* Read();
void Write(int x)
{write(x,msg1,MSGSIZE);
write(x,msg2,MSGSIZE);
write(x,msg3,MSGSIZE);
}
char* Read(int x)
{
read(x,inbuf,MSGSIZE);
return inbuf;
}


int main()
{

int p[2];
if(pipe(p)<0)
exit(1);
Write(p[1]);
for(i=0;i<3;i++)
{char* inbuf=Read(p[0]);
printf("%s\n",inbuf);
}

return 0;
}
