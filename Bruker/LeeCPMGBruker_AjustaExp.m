clear all; close all; clc
saved=1; %este valor 0 no guarda la data CPMG vs tiempo
savep=1; %este valor 0 no guarda el resultado de los ajustes

root='D:\datosBruker\190801_YTEC_X3_hum24hs\';
%root='D:\datosBruker\190801_YTEC\';
rootbg='D:\datosBruker\190801_YTEC_Background\'
ext='\fid';
figure(10);
for i=3:10;
dir=i;
j=i-2;%corrijo pq empieza de 3 el loop

n=num2str(dir);

infile=[root,n,ext];
outfile=[root,n];

infilebg=[rootbg,n,ext];

a=readbruker(infile);

sample=a.data;
samplesq=squeeze(sample);


b=readbruker(infilebg);

Bg=b.data;
Bgsq=squeeze(Bg);


res=samplesq-Bgsq;

% figure
% plot(real(samplesq))
% 
% plot(real(Bgsq),'k')
% 
% plot(real(res),'g')


TE=a.acq.d(6);
NE=length(sample);
t2list=linspace(TE,TE*NE,NE)*1000;
t2list=t2list';

figure(10)
hold on;
plot(real(res))
hold off;
res=res';
rescut=real(res)';
datass=[t2list,rescut];


%%%%%%%%%%%%%%%%%%guardo las cpmg decaimientos y tiempos%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if saved==1;
cd(outfile);
save('cpmg.dat','datass','-ascii');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%plot(t2list,real(res));



datass=[t2list,rescut];
     up = [inf;inf;inf;inf];       %limites los valores de t2 hay que escribirlos como -1/t2 en segundos)
     low = [-inf;-inf;-inf;-inf];
     st_ = [1;-0.1;3;-3.33];           % starting points
     
     fo_ = fitoptions('method','nonlinearleastsquares','Upper',up,'Lower',low,'maxiter',10e10);
     set(fo_,'startpoint',st_);
     [fiteo,gof,output]=fit(t2list,rescut,'exp2',fo_);   
        subplot(2,1,1)
        plot(fiteo,t2list,rescut)
        subplot(2,1,2)
        plot(output.residuals,'r')
              
      
        coeficientes=coeffvalues(fiteo);
        ci=confint(fiteo);
         area1=coeficientes(1,1);
        area2=coeficientes(1,3);
        t2a=-1*((coeficientes(1,2))^-1);
        t2b=-1*((coeficientes(1,4))^-1);
     result(j,1)=TE*1000;
        result(j,2)=area1;
        result(j,3)=t2a;
        result(j,4)=area2;
        result(j,5)=t2b;        
           tau(j)=TE*1000; 
           t2av(j)=t2a;
           t2bv(j)=t2b;
end
figure;
plot(tau,t2av,'-ob');
figure;
plot(tau,t2bv,'-ob');
        
cd(root)
if savep == 1
    save('DobleExponencial.dat','result','-ascii');
end