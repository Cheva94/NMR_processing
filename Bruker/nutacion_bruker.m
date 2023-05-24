a=readbruker(file);
np=a.acq.td(1);
n=a.acq.td(2);
odata=zeros(np,n);
x=linspace(1,np,np);
odata=a.data;
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

  hold off

   area(k)=sum(real(esp(2045:2055,k)));
   maximo=max(area);
   sumFid(k)=sum(real(datos(1:100,k)));
    
    t(k)=vp(k);
   end;  
 figure;
plot(t,area)

data=[t' area'/maximo];
datalog=[t' log10(area'/maximo)];
save('areaNorm.dat','data','-ascii');
save('logareaNorm.dat','datalog','-ascii');