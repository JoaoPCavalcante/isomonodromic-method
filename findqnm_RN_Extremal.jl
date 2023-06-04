#!/usr/bin/env julia
# -*- coding: utf-8 -*-
using ArbNumerics
using LinearAlgebra
using PyPlot
using PyCall
using SymPy
using Plots
pyplot()

setprecision(ArbComplex,digits=96)

spin = ArbReal(0) # negative for outgoing fields
angularl = ArbReal(0)

Nc = 64 # fixes level for convergents
Nf = 32 # fixes level for Fredholm expansion expansion
MAXSTEPS =200
								
function pieIII(th,sig)
    return  ArbNumerics.gamma(1-sig)^2*ArbNumerics.rgamma(1+sig)^2 *
            ArbNumerics.gamma(1+(th[2]+sig)/2)*ArbNumerics.rgamma(1+(th[2]-sig)/2) *
            ArbNumerics.gamma(1+(th[1]+sig)/2)*ArbNumerics.rgamma(1+(th[1]-sig)/2)
end

function invertseries(seq)
    result = zeros(ArbComplex,2,2,Nf+1)
    result[:,:,1] = [ 1 0; 0 1 ]
    for p = 2:Nf+1
        A = -seq[:,:,p]
        for q = 2:p-1
             A += -(seq[:,:,q]*result[:,:,p-q+1])
        end
        result[:,:,p] = A
    end
    return result
end

function geeIII(sig,th1,t0=1)
    psi = zeros(ArbComplex,2,2,Nf+1)
    a = (sig-th1)/2
    c = sig
    psi[:,:,1] = [ 1 0 ; 0 1 ]
    psi[:,:,2] = [ (a/c*t0) (-a/(c*(1+c))*t0) ; ((a-c)/(c*(1-c))*t0) (-(a-c)/c*t0) ]
    for p = 3:Nf+1
        psi[1,1,p] = ((a+p-2)/((c+p-2)*(p-1))*psi[1,1,p-1]*t0)
        psi[1,2,p] = ((a+p-2)/((c+p-1)*(p-2))*psi[1,2,p-1]*t0)
        psi[2,1,p] = ((a-c+p-2)/((-c+p-1)*(p-2))*psi[2,1,p-1]*t0)
        psi[2,2,p] = ((a-c+p-2)/((-c+p-2)*(p-1))*psi[2,2,p-1]*t0)
    end
    return psi
end

#=
function geeIII(sig,th1,t0=1)   #OLEG
    psi = zeros(ArbComplex,2,2,Nf+1)
    a = (sig-th1)/2
    c = sig
    psi[:,:,1] = [ 1 0 ; 0 1 ]
    psi[:,:,2] = [ (-(a-c)/c*t0) (-a/(c*(1+c))*t0) ; ((a-c)/(c*(1-c))*t0) (a/c*t0) ]
    for p = 3:Nf+1
        psi[1,1,p] = ((c-a+p-2)/((c+p-2)*(p-1))*psi[1,1,p-1]*t0)
        psi[1,2,p] = ((c-a+p-2)/((c+p-1)*(p-2))*psi[1,2,p-1]*t0)
        psi[2,1,p] = ((-a+p-2)/((-c+p-1)*(p-2))*psi[2,1,p-1]*t0)
        psi[2,2,p] = ((-a+p-2)/((-c+p-2)*(p-1))*psi[2,2,p-1]*t0)   
    end
    return psi
end
=#

function geeV(th1,th2,t0=1.0)
    psi = zeros(ArbComplex,2,2,Nf+1)
    a = (th1-th2)/2
    c = th1
    psi[:,:,1] = [ 1 0 ; 0 1 ]
    psi[:,:,2] = [ (a/c*t0) (-a/(c*(1+c))*t0) ; ((a-c)/(c*(1-c))*t0) (-(a-c)/c*t0) ]
    for p = 3:Nf+1
        psi[1,1,p] = ((a+p-2)/((c+p-2)*(p-1))*psi[1,1,p-1]*t0)
        psi[1,2,p] = ((a+p-2)/((c+p-1)*(p-2))*psi[1,2,p-1]*t0)
        psi[2,1,p] = ((a-c+p-2)/((-c+p-1)*(p-2))*psi[2,1,p-1]*t0)
        psi[2,2,p] = ((a-c+p-2)/((-c+p-2)*(p-1))*psi[2,2,p-1]*t0)
    end
    return psi
end

function buildAV(sig,thb)     # adicionado para calcular a tau_III
    vecg = geeIII(sig,thb)
    vecginv = invertseries(vecg)
    bigA = ArbNumerics.ArbComplexMatrix(zeros(ArbComplex,2Nf,2Nf))
    for p = 1:Nf
        for q = 1:Nf
            result = zeros(ArbComplex,2,2)
            if q + p <= Nf+1
                for r = 1:q
                    result += vecg[:,:,p+r]*vecginv[:,:,q-r+1]
                end
            end
            bigA[2*p-1:2*p,2*q-1:2*q] = result
        end
    end
    return bigA
end

function buildD(sig,ths,x,t0)
    vecg = geeV(sig,ths,t0)
    vecginv = invertseries(vecg)
    bigD = ArbNumerics.ArbComplexMatrix(zeros(ArbComplex,2Nf,2Nf))
    left = [ (1/x) 0 ; 0 1 ]
    right = [ (x) 0 ; 0 1 ]
    for p = 1:Nf
        for q = 1:Nf
            result = zeros(ArbComplex,2,2)
            if q + p <= Nf+1
                for r = 1:p
                    result += -vecg[:,:,p-r+1]*vecginv[:,:,q+r]
                end
            end
            bigD[2*p-1:2*p,2*q-1:2*q] = left*result*right
        end
    end
    return bigD
end

function BuildDdiff(sig,ths,x,t)
    vecg = geeV(sig,ths,t)
    vecginv = invertseries(vecg)
    bigD = ArbNumerics.ArbComplexMatrix(zeros(ArbComplex,2Nf,2Nf))
    left = [ (1/x) 0 ; 0 1 ]
    right = [ (x) 0 ; 0 1 ]
    sigma3 = [ 1 0 ; 0 -1 ]
    for p = 1:Nf
        for q = 1:Nf
            result = zeros(ArbComplex,2,2)
            if q + p <= Nf+1
                for r = 1:p
                    result += -vecg[:,:,p-r+1]*vecginv[:,:,q+r]
                end
            end
            result = (p+q-1)/t*result+sig/(2t)*(sigma3*result-result*sigma3)
            bigD[(2p-1):(2p), (2q-1):(2q)] = left*result*right
            end
    end
    return bigD
end

# The continuous fraction method below

function ei(th,sig,n)
    return (sig+th[3]+2*n-4)*(sig-th[3]-2*th[2]+2*n) +
           (2-th[3]-th[2])^2-th[1]^2
end

function bee(th,sig,n)
    return (sig+th[3]+2*n-2)*(sig-th[3]+2*n)
end

function see(th,sig,n)
    return 2*(sig+th[3]+2*n-2)
end

function dee(th,sig,n)
    return 2*(sig+th[3]+2*n)
end

function you0(th,sig,K,t)
    res = 0
    i = Nc
    while ( i > 0 )
        res = ei(th,sig,i)/(bee(th,sig,i)-4*K+t*(see(th,sig,i) -
              dee(th,sig,i)*res))
        i -= 1
    end
    return res
end

function vee0(th,sig,K,t)
    res = 0
    i = Nc
    while ( i > 0 )
        res = dee(th,sig,-i)/(bee(th,sig,-i)-4*K+t*(see(th,sig,-i) -
              ei(th,sig,-i)*res))
        i -= 1
    end
    return res
end

function equation(th,sig,K,t)
    return t*(ei(th,sig,0)*vee0(th,sig,K,t)-see(th,sig,0) +
           dee(th,sig,0)*you0(th,sig,K,t))+4*K-bee(th,sig,0)
end

function tauhatIII(th,sig,esse,t)
    x = esse*pieIII(th,sig)*t^sig
    A = buildAV(sig,th[2])
    D = buildD(-sig,th[1],x,t)   #th[1]= ths Está consistente, NÃO MEXER.
    Id = Matrix{ArbComplex}(I,2Nf,2Nf)
    Fred = ArbNumerics.ArbComplexMatrix(Id - A*D)
    return ArbNumerics.determinant(Fred)
end

function eiIII(th,sig,n)
    return 2*(sig+th[1]+2*n-4) + 4*(1-(th[1]+th[2])/2)
end

function beeIII(th,sig,n)      # não muda
    return (sig+th[1]+2*n-2)*(sig-th[1]+2*n)
end

function deeIII(th,sig,n)      # não muda
    return 2*(sig+th[1]+2*n)
end

function you0III(th,sig,K,t)  
    res = 0
    i = Nc
    while ( i > 0 )
        res = eiIII(th,sig,i)/(beeIII(th,sig,i)-4*K-t*(deeIII(th,sig,i)*res))
        i -= 1
    end
    return res
end

function vee0III(th,sig,K,t)    
    res = 0
    i = Nc
    while ( i > 0 )
        res = deeIII(th,sig,-i)/(beeIII(th,sig,-i)-4*K-t*(eiIII(th,sig,-i)*res))
        i -= 1
    end
    return res
end

function equationIII(th,sig,K,t)
    return t*(eiIII(th,sig,0)*vee0III(th,sig,K,t)+deeIII(th,sig,0)*you0III(th,sig,K,t))+4*K-beeIII(th,sig,0)
end


function muller(f,x,verb,tol=1e-20,maxsteps=MAXSTEPS::Int64)
    h = ArbFloat(1e-10)

	local xv = [ ball(ArbComplex(x))[1] ball(ArbComplex(x+h))[1] ball(ArbComplex(x-h))[1] ]
	local fv = [ ball(ArbComplex(f(xv[1])))[1] ball(ArbComplex(f(xv[2])))[1] ball(ArbComplex(f(xv[3])))[1] ]
	local counter = 1

	while (counter < maxsteps) && (ball(real(abs(xv[1] - xv[2])))[1] > tol) && (ball(real(abs(fv[1])))[1] > tol )
        divdiff = [ ((fv[1]-fv[2])/(xv[1]-xv[2])) ((fv[1]-fv[3])/(xv[1]-xv[3])) ((fv[2]-fv[3])/(xv[2]-xv[3])) ]
        ddivdiff = (divdiff[1]-divdiff[3])/(xv[1]-xv[3])
        doubleu = divdiff[1]+divdiff[2]-divdiff[3]

        discr = sqrt(doubleu^2-4*fv[1]*ddivdiff)
        if (ball(real(abs(doubleu-discr)))[1] > (ball(real(abs(doubleu+discr)))[1]))
            xnew = xv[1]-2*fv[1]/(doubleu-discr)
        else
            xnew = xv[1]-2*fv[1]/(doubleu+discr)
        end
        xv = [ ball(ArbComplex(xnew))[1] xv[1] xv[2] ]
        fv = [ ball(ArbComplex(f(xnew)))[1] fv[1] fv[2] ]
        if verb
            println(counter," ",Complex{Float64}(xv[1])," ",Float64(abs(fv[1])))
        end
	    counter += 1
	end

	if counter >= maxsteps
	    error("Did not converge in ", string(maxsteps), " steps")
    else
	    xv[1], counter
    end
end

function abs2d(pt)
    return ArbFloat(ArbNumerics.sqrt(pt[1]*conj(pt[1])+pt[2]*conj(pt[2])))
end

function newton2d(f,x,verb,tol=1e-20,maxsteps=MAXSTEPS::Int64)
    h = ArbFloat(1e-8)

    local counter = 0

    xnew = transpose(x)
    while true
        local xv = zeros(ArbComplex,3,2)
        local fv = zeros(ArbComplex,3,2)
        xv[1,:] = xnew
        xv[2,:] = xnew + [ h 0 ]
        xv[3,:] = xnew + [ 0 h ]
        for i = 1:3
            fv[i,:] = f(xv[i,:])
        end
        jac = [ ((fv[2,1] - fv[1,1])/h) ((fv[3,1] - fv[1,1])/h) ; ((fv[2,2] - fv[1,2])/h) ((fv[3,2] - fv[1,2])/h) ]
        step = transpose(Array{ArbComplex,1}(inverse(ArbNumerics.ArbComplexMatrix(jac))*fv[1,:]))
        xnew = first.(ball.(xnew-step))
        counter += 1
        println( counter," ",Complex{Float64}(xnew[1])," ",Complex{Float64}(xnew[2])," ",Float64(abs2d(fv[1,:])) )
        ((counter > maxsteps) || (abs2d(step) < tol) || (abs2d(fv[1,:]) < tol)) && break
    end

	if counter >= maxsteps
	    error("Did not converge in ", string(maxsteps), " steps")
    else
	    xnew, counter
    end
end

#Extremal angular and radial system
#=
function angulareqEXTREMAL(z,q)
    thang = zeros(ArbComplex,3)
    thang[1] = -angularm-spin
    thang[2] = angularm-spin
    thang[3] = -2.0*spin
    sigang = -(2.0*angularl+2.0)   # sigang -> -sigang for m<0
    tang = -4*z
    extrafac = 2*(2*spin-angularm+1)*z+z^2
    return equation(thang,sigang,q+extrafac,tang)
end
=#

function radialeqEXTREMAL(z,qQ)
    th = zeros(ArbComplex,3)
    th[1] = -2*spin-2im*(qQ-2*z)    #ths
    th[2] = 2*spin-2im*(qQ-2*z)     #thb
    zrad = -4*z*(z-qQ)
    extrafac = (angularl-spin)*(angularl+spin+1)-im*(1+2*spin)*(qQ-2*z)-4*z*(z-qQ)
    
    function func1(x)
        return equationIII(th,x,extrafac,-zrad)
    end
    sig0 = muller( func1, ArbComplex(1.1-0.1im), false )[1]
    #println(Complex{Float64}(sig0))
    ess0 =ArbNumerics.exp(-2im*pi*sig0)*(ArbNumerics.sinpi((th[1]+sig0)/2)*ArbNumerics.sinpi((th[2]+sig0)/2))/(ArbNumerics.sinpi((th[1]-sig0)/2)*ArbNumerics.sinpi((th[2]-sig0)/2))
    #println(Complex{Float64}(ess0))
    return tauhatIII(th,sig0,ess0,zrad)
end

function fr_wrapper(z)
    res = radialeqEXTREMAL(z,qQ)
    return res
end

#=
println("=============================================================================")

zinitial = ArbComplex(0.2463675811292359 - 0.10770887389007457im)
qQ = 0.217
solution = muller(fr_wrapper, zinitial, false )[1]
println("z = ",Complex{Float64}(solution))
println("Verify_z= ",Complex{Float64}(fr_wrapper(solution)))

println("=============================================================================")
=#


function rang(ini)
    i = ini
    #arq = open("/home/joaocavalcante/Dropbox/Painlevé V/RN Black Hole/Extremal RN/RNQnmExtremal.txt", "w")
    zinitial = ArbComplex(0.6557706014103134 - 0.038232448350358335im)
    while ( i < 1.001 )
        global qQ = i
	
	solution = ArbComplex(qQ - 0.0im)
	#solution = muller(fr_wrapper, zinitial, false )[1]
	#println("l = ",solution[1][2])
	#println("Verify_z= ", Complex{Float64}(fr_wrapper(solution)))
	#println(-4*solution[1][1]*(solution[1][1]-qQ)))
	println("qQ=",ArbFloat(qQ,digits=8),"\t","omega=", Complex{Float64}(solution))
	println(arq,ArbFloat(qQ,digits=8),",","\t",Complex{Float64}(solution[1][1]))
	#,",","\t", Complex{Float64}(-4*solution[1][1]*(solution[1][1]-qQ)) 
        zinitial = solution
        i += 0.001  #increment
    end
    return close(arq)
end

rang(0.216)


#=
qQ = 0.01

function density()
  x = [-2*pi + 4*pi*i/100 for i in 1:100]
  y = [-2*pi + 4*pi*i/100 for i in 1:100]
  z = [sin(x[i]) * cos(y[j]) * sin(x[i]*x[i]+y[j]*y[j])/log(x[i]*x[i]+y[j]*y[j]+1) for i in 1:100 for j in 1:100] 
  z_ = [z[i:i+99] for i in 1:100:10000]
  data = contour(;z=z_, x=x, y=y)
  plot(data)
end


density()


x = 1:0.5:20
y = 1:0.5:10
f(x, y) = begin
        (3x + y ^ 2) * abs(sin(x) + cos(y))
    end
X = repeat(reshape(x, 1, :), length(y), 1)
Y = repeat(y, 1, length(x))
Z = map(f, X, Y)
p1 = contour(x, y, f, fill = true)
p2 = contour(x, y, Z)
plot(p1, p2)
=#


