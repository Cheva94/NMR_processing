close all; clear all; clc;

%root='E:\DocumentosBEL\Datos\Bruker\151001_AdamantanoCentrado_DOTY\';

root='E:\DocumentosBEL\Datos\Bruker\Galo\EspectrosyDQ150910\150910_TiO_seco_DOTY';
folder='\3';
ext='\ser';
file=[root,folder,ext];
%%%%%%%%%%%%%%%%%Cargo parametros del experimento
a=readbruker(file);
n=a.acq.td(1);
n2=a.acq.td(2);
odata=zeros(n,n2);
zf=1;
odata=a.data;
vd=a.acq.vdel;
%vp=zeros(n,1);
%vp=a.acq.vp; %cuando variemos vp
np=n-69;
sw=a.acq.sw_h(1); % Escala en KHz
nf=np*zf;
ii=sqrt(-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:nf; %Armamos escala en KHz
    hz(i)=(i-1)*(sw/(nf-1))-sw/2;
    ppm(i)=hz(i)/300;
end

%fil=apod('g',(np),0,100);  %tipo de filtro, dimension, centro, ancho 
% fil=fil';
 %figure;
for k=1:n2
%for k=7:7
    datos(1,k)=odata(70,k)/2;
    for i=71:n
    datos(i-69,k)=odata(i,k);
    end
 %   datafil(:,k)=datos(:,k).*fil;
    
    %Phase Correction
     for j=1:360
    aa=2*pi/360;
    %data_c(:,k)=datafil(:,k)*exp(ii*j*aa);
    data_c(:,k)=datos(:,k)*exp(ii*j*aa);
    area(j,k)=real(data_c(1,k));
     end
   [val,index]=max(area(:,k));
   data_ph(:,k)=datos(:,k)*exp(ii*index*aa);
    %%%%%%%%%%
    
    %Espectros con correcciones
    esp(:,k)=(fftshift((fft(datos(:,k),np*zf))));  
   % espfil(:,k)=(fftshift((fft(datafil(:,k),np*zf)))); 
    espph(:,k)=(fftshift((fft(data_ph(:,k),np*zf)))); 
    %%%%%%%%%%%%%
%Ploteamos las señales para definir filtro-    
%  figure; 
%  hold on;
%  plot(real(data_ph(:,k))/max(real(datos(:,k))));
%  plot(imag(datos(:,k))/max(real(datos(:,k))),'g-');
%  plot(real(datafil(:,k))/(max(real(datos(:,k)))),'m-')
%  plot(fil,'r-')
%  hold off;

%ploteo de espectros y chequeo de corrección de Fase
%
%hold on;
%plot(hz,real(espph(:,k))); %plot en escala en hz
 %figure;
%plot(real(data_c(:,k)));
%figure;
%hold on;
%plot(real(espph(:,k)));%plot en numero de puntos para definir region a sumar
%hold off;

areat(k)=sum(real(espph(895:1100,k)))/(1100-895);
%areaesp_c(k)=sum(real(espph(931:995,k)))/8; %Señal en el centro que coincide con la REF
 areaesp_i(k)=sum(real(espph(884:961,k)))/(961-884); %Pico ancho a la izquierda
 areaesp_d(k)=sum(real(espph(1020:1098,k)))/(1098-1020); %Pico ancho a la derecha
 sumFid(k)=sum(real(data_ph(1:10,k)))/10;
 sumFid2(k)=sum(real(datos(1:10,k)))/10;
    td(k)=vd(k);
    %tp(k)=vp(k) %para listas vp
   end; 
   figure;
    plot(real(espph(:,8)));
maxFid=max((sumFid));
maxFid2=min((sumFid2));

maxArea=max(areat);
%maxArea_c=max(areaesp_c);
maxArea_i=max(areaesp_i);
maxArea_d=max(areaesp_d);
figure;
hold on;
% plot(td,sumFid/maxFid);
% plot(td,(areat/maxArea),'m*');
% plot(td,(areaesp_i/maxArea_i),'r');
% plot(td,(areaesp_d/maxArea_d),'go');
 plot(td,sumFid);
plot(td,(areat),'m*');
plot(td,(areaesp_i),'r');
plot(td,(areaesp_d),'go');
hold off;

data=[td' sumFid' areat' areaesp_i' areaesp_d'];
%dataAncha=[td' areaesp_i'];

%datan=[td' (sumFid/maxFid)' (areat/maxArea)' (areaesp_i/maxArea_i)'];
%datitos=[ppm',real(espph(:,10))];
%figure; 
%hold on;
%  plot(real(data_ph(:,10))/max(real(data_ph(:,10))));
%  plot(real(datos(:,10))/max(real(datos(:,10))),'g-');
% plot(real(datafil(:,10))/(max(real(datafil(:,10)))),'m-')
%  plot(fil,'r-')
%  hold off;
%  

%datalog=[t' log10(area'/maximo)];
%save('DataProc.dat','datitos','-ascii');
%save('DQancha.dat','dataAncha','-ascii');
save('DQnoviembre.dat','data','-ascii');
%save('logareaNorm.dat','datalog','-ascii');
  