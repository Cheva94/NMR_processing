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
