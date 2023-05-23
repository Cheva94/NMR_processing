close all; clear all; clc;
% function result=buildup(varargin)
% if nargin~=0
%   if exist(varargin{:},'file')==2
%      fname = varargin{:};
%   else
%       disp('ERROR: file not found..please select now!');  
%       [filename,dirname]=uigetfile('*.*','Abrir archivo ser');
%       fname=[dirname,filename];
%       cd dirname;
%   end
% else
% [filename,dirname]=uigetfile('*.*','Abrir archivo ser');
% fidname=[dirname,filename];
% cd(dirname)
% end

%a=readbruker(fidname);
root='D:\datosBruker\200219_Zn22';
%root='E:\DocumentosBEL\Datos\Bruker\140903_H2O_CuSO';
folder='\2';
ext='\ser';
file=[root,folder,ext];
folderb='\3';
fileb=[root,folderb,ext];
a=readbruker(file);
ab=readbruker(fileb);
np=a.acq.td(1);
n=a.acq.td(2);
odata=zeros(np,n);
odatab=zeros(np,n);
x=linspace(1,np,np);
odata=a.data;
odatab=ab.data;
odata=odata-odatab;
zf=1;
vp=zeros(n,1);
 vp=a.acq.vp;
for k=1:n
        for i=71:np
    datos(i-70,k)=odata(i,k);
    end
    esp(:,k)=(fftshift((fft(datos(:,k),np*zf))));
    
   hold on
  plot(abs(esp(:,k)));
  % plot(real(datos(:,k)));
  hold off
    % plot(real(datos(:,k)));
    
     
   %maximo=max(esp);
   area(k)=sum(real(esp(2045:2055,k)));
   maximo=max(area);
   sumFid(k)=sum(real(datos(1:100,k)));
    %integral(k)=area;
    %data(k)=real(fid);
    %nutac(k)=data(k);%/ref;
    t(k)=vp(k);
   end;  
 figure;
plot(t,area)


% i=1;
% while arr(i) < sin(pi/3)
%     i=i+1;
% end;
% %suponiendo una relacion lineal entre t(i) y t(i+1)
% pi3= (t(i+1)-t(i))/(arr(i+1)-arr(i))*(sin(pi/3)-arr(i))+t(i); 
% i=1;
% flag=1;
% while arr(i) < sin(pi/4)
%     i=i+1;
%     flag=0;
% end;
% if flag == 1
%     text(t(end)/2,0.2,['No hay datos suficientes para  \pi /4 ']);
% end;
% %suponiendo una relacion lineal entre t(i) y t(i+1)
% pi4= (t(i+1)-t(i))/(arr(i+1)-arr(i))*(sin(pi/4)-arr(i))+t(i); 
% 
% text(t(end)/2,0.4,[' \pi /2 =  ',num2str(t(ii)),' \mu s']);
% text(t(end)/2,0.3,[' \pi /3 =  ',num2str(pi3),' \mu s']);
% if flag == 0
%     text(t(end)/2,0.2,[' \pi /4 =  ',num2str(pi4),' \mu s']);
% end;

%data=[t' sumFid' area'];

data=[t' area'/maximo];
datalog=[t' log10(area'/maximo)];
save('areaNorm.dat','data','-ascii');
save('logareaNorm.dat','datalog','-ascii');
  