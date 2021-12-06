Acá pongo la función flint1d que llamo en el programa de procesamiento
***********************************
function [S,resida] = flint1D(K1,Z,alpha,S)
%Version1D a partir de
% Fast 2D NMR relaxation distribution estimation - Matlab/octave version
% Paul Teal, Victoria University of Wellington
% paul.teal@vuw.ac.nz
% Let me know of feature requests, and if you find this algorithm does
% not perform as it should, please send me the data-set, so I can improve it.
% Issued under the GNU AFFERO GENERAL PUBLIC LICENSE Version 3.
% If you distribute this code or a derivative work you are required to make the
% source available under the same terms
% If you use this software, please cite P.D. Teal and C. Eccles. Adaptive
% truncation of matrix decompositions and efficient estimation of NMR
% relaxation distributions. Inverse Problems, 31(4):045010, April
% 2015. http://dx.doi.org/10.1088/0266-5611/31/4/045010 (Section 4: although
% the Lipshitz constant there does not have alpha added as it should have)

% Y is the NMR data for inversion
% alpha is the (Tikhonov) regularisation (scalar)
% S is an optional starting estimate

% K1 is the kernel matrices
% They can be created with something like this:
maxiter = 100000;

if nargin<4
  Nx = size(K1,2);  % N1 x Nx
   S = ones(Nx,1);  % initial estimate
end

if nargout>1
  resida = NaN(maxiter,1);
end

KK1 = K1'*K1;
KZ12 = K1'*Z;

% Lipschitz constant
L = 2 * (trace(KK1) + alpha); % trace will be larger than largest
                                     	% eigenvalue, but not much larger
   		    
tZZ = trace(Z*Z');   	% used for calculating residual

Y = S;
tt = 1;
fac1 = (L-2*alpha)/L;
fac2 = 2/L;
lastres = inf;

for iter=1:maxiter
  term2 = KZ12-KK1*Y;
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
	%resid = tZZ -2*trace(S'*KZ12) + trace(S'*KK1) + normS;
	resid = tZZ -2*(S'*KZ12) + (S'*KK1*S) + normS;
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

************************************

Ahora pongo el código para procesamiento de datos 1d tiene varias cosas comentadas que se usan depende lo que estemos procesando.

clear all;close all;
cd('C:\Users\belen\Google Drive\datos\minispec\Background_Feb19\');
load MSE_CPMG.txt

bg=MSE_CPMG(161:310,2);
save('MSE_BackGdata.dat','bg','-ascii')
cd('C:\Users\belen\Google Drive\datos\minispec\Almidon06Feb2019\Muestra4\');
%load data.dat
load MSE_CPMG_2.txt
d=MSE_CPMG_2(161:310,2);
save('MSE_CPMGdata.dat','d','-ascii')
tau1=MSE_CPMG_2(161:310,1);
save('MSE_CPMGtau.dat','tau1','-ascii')
Z=d-0*bg;
save('MSE_CPMG_restaBG.dat','Z','-ascii')
%load tiempo.dat
%tau1=tiempo';
alpha=1E-9;
Nx = 500;  	% number of bins in relaxation time grids
T1 = logspace(-1.5,2.5,Nx);  %T2
d=length(tau1);
for i=1:d
K1(i,:) = exp(-tau1(i) *(1./T1) );  % T1 relaxation data   T1f
end
 [S,resida] = flint1D(K1,Z,alpha);
figure
 semilogx(T1,S')
 T1=T1';
 datos=[T1,S];

%save('ILT1DCPMG.dat','S','-ascii');
xlabel('T_2') % x-axis label

%save('TiemposmseCPMG.dat','datos','-ascii');
%  






