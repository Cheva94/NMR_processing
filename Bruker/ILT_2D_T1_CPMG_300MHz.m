
clear all; clc

root='D:\googleDrive\datos\Del300\201009_C1C2\';
folder='15';
%folderb='18';
ext='\ser';

file=[root,folder,ext];
%fileb=[root,folderb,ext];


%******************************************************
%  Configura los parametros para la Transf. de Laplace
%******************************************************
nini=1;
niniT1=1;
alpha=1E-1;

Nx = 100;      % number of bins in relaxation time grids
Ny = 100;

T1 = logspace(-3,4,Ny);  %T1  completo (-3,3,Ny)
T2 = logspace(-2,1,Nx);  %T2  completo (-3,3,Ny)

%************************************************
%      Lee los datos y los pone en fase
%************************************************
sve = 1; % 1 saves plots

a=readbruker(file);
datas=a.data;
echoes=a.acq.td(1);
dimInd=a.acq.td(2);
datasresh=reshape(datas,echoes,dimInd); %6016 o 3072

%b=readbruker(fileb);

%dataB=b.data;
%datasreshB=reshape(dataB,6016,50);

data=datasresh%-datasreshB;

time1=a.acq.vdel;
DW=a.acq.d(6)*1000;
time1=time1';
time1=time1(1:50);

for i=1:echoes;
    time2(i)=DW*i;
end

time2=time2';

figure
surf(time1,time2,real(data))


%for i=1:360
%ph(i)=i;
%data_c(:,i)=data(:,50)*exp(1i*ph(i)*pi/180);
%pim(i)=sum(imag(data_c(:,i)));
%pre(i)=sum(real(data_c(:,i)));
%end


%[ma,in_ma]=max(pre);

for j=1:360
    aa=pi/360;
   data_c=data(:,dimInd)*exp(1i*j*aa);
    area(j)=real(data_c(1));
end
   [val,index]=max(area);
   datac=data*exp(1i*index*aa);

%datac=data*exp(1i*pi*in_ma/180);

figure
plot(real(datac(:,5)),'b');
hold on
plot(imag(datac(:,5)),'r');
hold off
title('Señal puesta en fase');
datafin=real(datac);

figure
surf(time1(niniT1:end),time2(nini:end),datafin(nini:end,niniT1:end))
% figure
% plot(time2(nini:end),datafin(nini:end,end),'bo')

% %************************************************
% %      Resta el background
% %************************************************
% % 
%   cd ('D:\Data\500MHz_YTEC\Soil 1 1 1A Background\')
%   load('background1A.dat');
%   
%  back=background1A; 
%  %dataout=(datafin-back)/3.29e+04;
%  dataout=(datafin-back);
% 
%  %************ Grafico fase ***************
% figure(1)
% surf(real(dataout));
% title('Señal puesta en fase');

%************* Inversion *********************

Z=datafin(nini:end,niniT1:end);
Z=Z';
tau2=time2(nini:end);
tau1=time1(niniT1:end);
% tau1=tau1';
% figure
% surf(tau1,tau2,real(datac))

%******* Elegir el kernel, dependiendo si la medicion fue con IR o SR
%-----------------------------------------------------------------
% K1 = 1-2*exp(-tau1*(1./T1) );  % T1-T2 (IR)
K1=1-exp(-tau1*(1./T1));  % T1-T2 (SR)
K2 = exp(-tau2 *(1./T2) );  % T2 relaxation data
%-----------------------------------------------------------------
%*******

[S,resida] = flint(K1,K2,Z,alpha); %Transformada ILT en 2D
figure(11)
contour(T2,T1,S,90)
set(gca,'YScale','log','FontSize',13)
set(gca,'XScale','log','FontSize',13)
xlabel('T_{2} [ms]','FontSize',18)
ylabel('T_{1} [ms]','FontSize',18)
% caxis([0 150])
colorbar

% yy=ones(Ny)*10;
% hold on;
% %x = [1e-3 10];
% %y = [10 10];
% %line(x,y,'Color','red','LineStyle','--')
% x1 = [1e-2 1e4];
% x2000 = [1e-2 1e4/2000];
% x300 = [1e-2 1e4/300];
% line(x1,x1,'Color','red','LineStyle','-', 'LineWidth',2)
% line(x2000,2000*x2000,'Color','red','LineStyle','--', 'LineWidth',2)
% line(x300,300*x300,'Color','red','LineStyle',':', 'LineWidth',2)

hold off;
if sve == 1
    saveas(gcf,'contour_Soil2.jpg');
end
%------------ PROYECCIONES -------------
T1=T1';
T2=T2';

aa=sum(S);
aa=aa';        % Proyeccion de T2
St=S';
b=sum(St);
b=b';        % Proyeccion de T1


% %------------ CUMULATIVO -------------
for i=1:length(aa)
    cT2(i)=sum(aa(1:i));
end
for i=1:length(b)
    cT1(i)=sum(b(1:i));
end
% *************************

% % *************************
%
figure(30)
plot(T2,aa)
hold on
plot(T2,cT2/max(cT2)*max(aa),'r');
set(gca,'XScale','log')
xlabel('T_{2} [ms]')
hold off

figure(40)
plot(T1,b)
hold on
plot(T1,cT1/max(cT1)*max(b),'r');
set(gca,'XScale','log')
xlabel('T_{1} [ms]')
hold off

%**** Calcula el cociente signal to noise ******
d=length(tau2);
sig=sqrt(mean(Z(end,1:2).^2));
N=sqrt(mean(Z(end,d-round(d/10):d).^2));
SnR=sig/N

%***********************************************

Szero=S;
% Szero(1:100,1:20)=0;
% Szero(1:10,1:100)=0;
% Szero(1:100,1:30)=0;
% Szero(85:100,1:100)=0;
% figure
% imagesc(Szero)


figure
set(gcf,'units','points','position',[100,100,678.6,600])
contour(T2,T1,Szero,150,'LineWidth',2)
set(gca,'YScale','log','FontSize',20,'LineWidth',2)
set(gca,'XScale','log','FontSize',20,'LineWidth',2)
xlabel('T_{2} [ms]','FontSize',25)
ylabel('T_{1} [ms]','FontSize',25)
% caxis([0 0.14])
colorbar
sumtotal=sum(sum(Szero))
return
[d1,k1]=size(tau1);
[d2,k2]=size(tau2);

dim1=length(T1); % Tamaño del mapa
dim2=length(T2);

M=zeros(d1,d2);
for k=1:d1    
    %k
    for i=1:d2     
        for kk=1:dim1
            for ii=1:dim2              
                R1=(1-exp(-tau1(k)/T1(kk)));
                R2=exp(-tau2(i)/T2(ii));
                M(k,i)=M(k,i)+S(kk,ii)*R2*R1;
            end
        end
    end
end
M=M';
figure
surf(tau1,tau2,M)

dataout=datafin(nini:end,:);

% Cálculo de r cuadrado para ver qué tan bueno es el ajuste
% FID-CPMG
yresid = dataout(end,:) - M(end,:);
SSresid = sum(yresid.^2);
SStotal = (length(dataout(end,:))-1) * var(dataout(end,:)); 
rsq = 1 - SSresid/SStotal;
% T1 primer punto
yresidT1 = dataout(:,1) - M(:,1);
SSresidT1 = sum(yresidT1.^2);
SStotalT1 = (length(dataout(:,1))-1) * var(dataout(:,1)); 
rsqT1 = 1 - SSresidT1/SStotalT1;

Nrescpmg=sqrt(mean(yresid.^2));
SnRrescpmg=sig/Nrescpmg

NresT1=sqrt(mean(yresidT1.^2));
SnRresT1=sig/NresT1


figure(201)
subplot(2,1,1)
plot(tau2,dataout(end,:),'bo')
title('FID-CPMG T1 saturado');
hold on
plot(tau2,M(end,:),'r')
annotation('textbox', [0.25, 0.8, 0.1, 0.1], 'String', "r^2=" + num2str(rsq),'FontSize',12)
axes('Position',[.62 .65 .27 .27])
box on
plot(tau2(1:12),dataout(end,1:12),'bo')
hold on
plot(tau2(1:12),M(end,1:12),'r')
legend('Data','Ajuste')


subplot(2,1,2)
plot(tau1,dataout(:,2),'bo')
title('T1 primer punto de la FID');
hold on
plot(tau1,M(:,2),'r')
legend('Data','Ajuste')
annotation('textbox', [0.25, 0.15, 0.1, 0.1], 'String', "r^2=" + num2str(rsqT1),'FontSize',12)



return

save('T1T2mapShaleIFC.dat','S','-ascii');
save('T1.dat','T1','-ascii');
save('T2.dat','T2','-ascii');

T2r=[T2 aa];
T1r=[T1 b];
cT1r=[T1 cT1'];
cT2r=[T2 cT2'];

save('proy_T2.dat','T2r','-ascii');
save('proy_T1.dat','T1r','-ascii');
save('tau1.dat','tau1','-ascii');
save('tau2.dat','tau2','-ascii');
save('datafin.dat','datafin','-ascii');
save('cumulativo_T1.dat','cT1r','-ascii');
save('cumulativo_T2.dat','cT2r','-ascii');

%***********************************************
c=size(S);
b=zeros(c(1),c(1));

% % SOIL 1A a 2 MHz
A(1,:)=[15,55,10,38]; %    A
A(2,:)=[55,90,10,38]; %   B
A(3,:)=[30,55,38,85]; %  C
A(4,:)=[55,90,38,85]; %   D

for i=1:4
    % define rangos
    b(A(i,1):A(i,2),A(i,3))=1;
    b(A(i,1):A(i,2),A(i,4))=1;
    b(A(i,1),A(i,3):A(i,4))=1;
    b(A(i,2),A(i,3):A(i,4))=1;
    
    % calcula el area del pico
    amp(i)= sum(sum(S(A(i,1):A(i,2),A(i,3):A(i,4))));
    
    % mascara para mostrar el pico aislado
    pdata=zeros(c(1),c(1));
    mask=zeros(c(1),c(1));
    mask(A(i,1):A(i,2),A(i,3):A(i,4))=1;
    
    pdata=S.*mask;
    
    %Proyecciones en cada eje del pico
    p2(i,:)=sum(pdata);
    pdata=pdata';
    p1(i,:)=sum(pdata);
    
end


%--- Inversión de la laplace ----
% Para reconstruir la matriz de datos a partir de la inversión del mapa
% de la ILT se usan los mismos ejes de tiempos con el que se midió. 


[d1,k1]=size(tau1);
[d2,k2]=size(tau2);

dim1=length(T1); % Tamaño del mapa
dim2=length(T2);

M=zeros(d1,d2);
for k=1:d1    
    %k
    for i=1:d2     
        for kk=1:dim1
            for ii=1:dim2              
                R1=(1-exp(-tau1(k)*T1(kk)));
                R2=exp(-tau2(i)/T2(ii));
                M(k,i)=M(k,i)+S(kk,ii)*R2*R1;
            end
        end
    end
end
M=M;
figure(2)
surf(tau2,tau1,M)

dataout=datafin';
% Cálculo de r cuadrado para ver qué tan bueno es el ajuste
% FID-CPMG
yresid = dataout(end,:) - M(end,:);
SSresid = sum(yresid.^2);
SStotal = (length(dataout(end,:))-1) * var(dataout(end,:)); 
rsq = 1 - SSresid/SStotal;
% T1 primer punto
yresidT1 = dataout(:,1) - M(:,1);
SSresidT1 = sum(yresidT1.^2);
SStotalT1 = (length(dataout(:,1))-1) * var(dataout(:,1)); 
rsqT1 = 1 - SSresidT1/SStotalT1;


figure(3)
subplot(2,1,1)
plot(tau2,dataout(end,:),'bo')
title('FID-CPMG T1 saturado');
hold on
plot(tau2,M(end,:),'r')
annotation('textbox', [0.25, 0.8, 0.1, 0.1], 'String', "r^2=" + num2str(rsq),'FontSize',12)
axes('Position',[.62 .65 .27 .27])
box on
plot(tau2(1:12),dataout(end,1:12),'bo')
hold on
plot(tau2(1:12),M(end,1:12),'r')
legend('Data','Ajuste')


subplot(2,1,2)
plot(tau1,dataout(:,1),'bo')
title('T1 primer punto de la FID');
hold on
plot(tau1,M(:,1),'r')
legend('Data','Ajuste')
annotation('textbox', [0.25, 0.15, 0.1, 0.1], 'String', "r^2=" + num2str(rsqT1),'FontSize',12)



return


%Calcula el decaimiento


%---------------- CUMULATIVO -------------------------------------
for j=1:Nx
    acc(j)=sum(S(1:j));
end
%-----------------------------------------------------------------
subplot(2,2,2)
plot(tau1,Z,'bo');
hold on;
plot(tau1,M,'r-');
%text(d/6,Z(1)/2,['SNR = ',num2str(round(SnR))],'Color','red','FontSize',18)
hold off
xlabel('t (ms)');
title(['SNR = ',num2str(round(SnR))]);
subplot(2,2,3)
semilogx(T,S')
hold on
yyaxis left

ylabel('Distribucion');

yyaxis right
%semilogx(T,acc/max(acc)*max(S),'r');
semilogx(T,acc,'r');
ylabel('Cumulativo (p.u.)');

hold off
xlabel('T_2 (ms)');

subplot(2,2,4)
plot(time,Res,'ko-')
xlabel('t (ms)');
ylabel('Residuo');

% t_ind=a.acq.vdel;
% t_ind=t_ind';
% t_dir=t_dir';
% t_dir=t_dir*1000;
% 
% save('C:\Yami\YTEC\Ca\mdataSRcpmg.dat','mdata','-ascii');
% save('C:\Yami\YTEC\Ca\tSRcpmg_dir.dat','t_dir','-ascii');
% save('C:\Yami\YTEC\Ca\tSRcpmg_ind.dat','t_ind','-ascii');
% figure
% [X,Y]=meshgrid(t_ind,t_dir);
% surf(X,Y,mdata)
% figure
% plot(t_ind,mdata(1,:),'bo-')