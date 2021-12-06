function [S,resida] = flint(K1,K2,Z,alpha,S)

% K1 and K2 are the kernel matrices
% They can be created with something like this:
%N1 = 50;   	% number of data points in each dimension
%N2 = 10000;
%Nx = 100;  	% number of bins in relaxation time grids
%Ny = 101;
%tau1min = 1e-4;
%tau1max = 10;
%deltatau2 = 3.5e-4;
%T1 = logspace(-2,1,Nx);
%T2 = logspace(-2,1,Ny);
%tau1 = logspace(log10(tau1min),log10(tau1max),N1)';
%tau2 = (1:N2)'*deltatau2;
%K2 = exp(-tau2 * (1./T2) ); 	% simple T2 relaxation data
%K1 = 1-2*exp(-tau1 *(1./T1) );  % T1 relaxation data

maxiter = 100000;

if nargin<5
  Nx = size(K1,2);  % N1 x Nx
  Ny = size(K2,2);  % N2 x Ny
  S = ones(Nx,Ny);  % initial estimate
end

if nargout>1
  resida = NaN(maxiter,1);
end

KK1 = K1'*K1;
KK2 = K2'*K2;
KZ12 = K1'*Z*K2;

% Lipschitz constant
L = 2 * (trace(KK1)*trace(KK2) + alpha); % trace will be larger than largest
                                     	% eigenvalue, but not much larger

tZZ = trace(Z*Z');   	% used for calculating residual

Y = S;
tt = 1;
fac1 = (L-2*alpha)/L;
fac2 = 2/L;
lastres = inf;

for iter=1:maxiter
  term2 = KZ12-KK1*Y*KK2;
  Snew = fac1*Y + fac2*term2;
  Snew = max(0,Snew);

  ttnew = 0.5*(1 + sqrt(1+4*tt^2));
  trat = (tt-1)/ttnew;
  Y = Snew + trat * (Snew-S);
  tt = ttnew;
  S = Snew;

  if ~mod(iter,500)
	% Don't calculate the residual every iteration; it takes much longer
	% than the rest of the algorithm
	normS = alpha*norm(S(:))^2;
	resid = tZZ -2*trace(S'*KZ12) + trace(S'*KK1*S*KK2) + normS;
	if nargout>1
  	resida(iter) = resid;
	end
	resd = abs(resid-lastres)/resid;
	lastres = resid;
	% show progress
	fprintf('%7i % 1.2e % 1.2e % 1.2e % 1.4e % 1.4e \n',...
      iter,tt,trat,L,resid,resd);
	if resd<1e-5
  	return;
	end
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Acá un código que levanta los datos del mini y hace la transformada
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%******************************************************
%  Configura los parametros para la Transf. de Laplace
%******************************************************
nini=1;
niniT1=1;
alpha=5E0;

Nx = 100;  	% number of bins in relaxation time grids
Ny = 100;

T1 = logspace(-3,4,Ny);  %T1  completo (-3,3,Ny)
T2 = logspace(-3,3,Nx);  %T2  completo (-3,3,Ny)

%************************************************
%  	Lee los datos y los pone en fase
%************************************************

data=load('211007_c6-conf-at-d3-void_SR-CPMG_mediahora.txt');
tau1=load('211007_c6-conf-at-d3-void_SR-CPMG_mediahora_t1.dat');
tau2=load('211007_c6-conf-at-d3-void_SR-CPMG_mediahora_t2.dat');

N=length(tau1);
M=length(tau2);

dataout=reshape(data(:,1),M,N);
% dataout=dataout(nini:end,niniT1:end);

%************* Inversion *********************

Z=dataout';

K1=  1-exp(-tau1*(1./T1));  % T1-T2 (SR)
K2 = exp(-tau2 *(1./T2) );  % T2 relaxation data

[S,resida] = flint(K1,K2,Z,alpha); %Transformada ILT en 2D
%-----------------------------------------------------------------


%************* Ploteo *********************

figure(2)
contour(T2,T1,S,90)
set(gca,'YScale','log','FontSize',13)
set(gca,'XScale','log','FontSize',13)
xlabel('T_{2} [ms]','FontSize',18)
ylabel('T_{1} [ms]','FontSize',18)
% caxis([0 0.1])
colorbar

yy=ones(Ny)*10;
hold on;
x1 = [1e-3 1e3];
x12 = [1e-3 1e3/12];
x900 = [1e-3 1e3/900];
line(x1,x1,'Color','red','LineStyle','-', 'LineWidth',2)
line(x12,12*x12,'Color','red','LineStyle','-.', 'LineWidth',2)
line(x900,900*x900,'Color','red','LineStyle',':', 'LineWidth',2)
legend('Data','T_{1}=T_{2}','T_{1}=12T_{2}','T_{1}=900T_{2}')
hold off;




%------------ PROYECCIONES -------------
T1=T1';
T2=T2';

aa=sum(S);
aa=aa';    	% Proyeccion de T2
St=S';
bb=sum(St);
bb=bb';    	% Proyeccion de T1


% %------------ CUMULATIVO -------------
for i=1:length(aa)
	cT2(i)=sum(aa(1:i));
end
for i=1:length(bb)
	cT1(i)=sum(bb(1:i));
end
% *************************

% % *************************
% %
figure(3)
plot(T2,aa)
hold on
plot(T2,cT2/max(cT2)*max(aa),'r');
set(gca,'XScale','log')
xlabel('T_{2} [ms]')
hold off

figure(4)
plot(T1,bb)
hold on
plot(T1,cT1/max(cT1)*max(bb),'r');
set(gca,'XScale','log')
xlabel('T_{1} [ms]')
hold off

T2r=[T2 aa];
T1r=[T1 bb];
cT1r=[T1 cT1'];
cT2r=[T2 cT2'];

t1=T1r(:,1);
t2=T2r(:,1);
time1=log10(t1);
time2=log10(t2);
[X,Y]=meshgrid(time2,time1);

%**** Calcula el cociente signal to noise ******
d=length(tau2);
sig=sqrt(mean(Z(end,1:2).^2));
N=sqrt(mean(Z(end,d-round(d/10):d).^2));
SnR=sig/N;

%***********************************************
c=size(S);
b=zeros(c(1),c(1));

% % SOIL 1A a 2 MHz
A(1,:)=[15,44,1,35]; %	A
A(2,:)=[44,90,1,35]; %   B
A(3,:)=[25,50,35,85]; %  C
A(4,:)=[50,90,35,85]; %   D

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

figure(5);
subplot(2,1,1)
contour(S,90);
hold on;
contour(b,'r');
hold off;
subplot(2,1,2)
contour(X,Y,S,90);
xlabel('log10(T2) (ms)');
ylabel('log10(T1) (ms)');
proy1=sum(p1);
proy2=sum(p2);


p1=p1';
p2=p2';
proy1=proy1';
proy2=proy2';
amp=amp';
%  T1T2=T1T2';

p1_out=[t1,p1,proy1];
p2_out=[t2,p2,proy2];



return


save('Proy_ind_T1.dat','p1_out','-ascii');
save('Proy_ind_T2.dat','p2_out','-ascii');
save('Amplitudes.dat','amp','-ascii');
% save('T1T2.dat','T1T2','-ascii');

save('T1T2mapShaleIFC.dat','S','-ascii');
save('T1.dat','T1','-ascii');
save('T2.dat','T2','-ascii');

T2r=[T2 aa];
T1r=[T1 bb];
cT1r=[T1 cT1'];
cT2r=[T2 cT2'];
