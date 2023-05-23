close all; clear all; clc;

root='C:\Users\belen\Google Drive\datos\Del300\Zn22\';
%root='E:\DocumentosBEL\Datos\Bruker\Galo\EspectrosyDQ150910\150910_TiO_seco_DOTY\';

%root='E:\DocumentosBEL\Datos\Bruker\140903_H2O_CuSO';
folder='1_rotando';
ext='\fid';
file=[root,folder,ext];
a=readbruker(file);
n=a.acq.td(1);
odata=a.data;
zf=1;
%datos(1)=odata(70)/2;
np=n-20%-69;
sw=a.acq.sw_h; % Escala en Hz
nf=np*zf;
ii=sqrt(-1);

for i=71:n  %Saco los primerps puntos
    datos(i-69)=odata(i);
end
datos(1)=2*odata(70);
for i=1:nf; %Armamos escala en KHz
    hz(i)=(i-1)*(sw/(nf-1))-sw/2;
    ppm(i)=hz(i)/300;
end
%fil=apod('g',(np),0,900);  %tipo de filtro, dimension, centro, ancho  
%fil=fil;
fil=1;
datafil=datos.*fil;

%Phase Correction
for k=1:360
    aa=2*pi/360;
    data_c=datafil*exp(ii*k*aa);
    area(k)=real(data_c(1));
end
[val,index]=max(area);
data_ph=datos*exp(ii*index*aa);
 %%%%%%%%%%
esp=(fftshift((fft(datos,np*zf))));  
espfil=(fftshift((fft(datafil,np*zf)))); 
espph=(fftshift((fft(data_ph,np*zf)))); 
figure;
hold on;
plot(real(datos)/max(real(datos)));
plot(imag(datos)/max(real(datos)),'b-');
plot(real(datafil)/(max(real(datos))),'m-')
plot(fil,'r-')
hold off;
%hold on
figure;
hold on;
plot(real(esp));
plot(((real(espph))),'m-');
plot(imag(esp),'g');
hold off;
datitos=[ppm' real(espph)' imag(espph)'];

%save('dataProc.dat','datitos','-ascii');


  