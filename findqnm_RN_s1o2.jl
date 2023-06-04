#!/usr/bin/env julia
# -*- coding: utf-8 -*-
using ArbNumerics
using LinearAlgebra
using PyPlot
using PyCall

setprecision(ArbComplex,digits=256)

# modes. Perhaps this needs a small fractional part...
spin = ArbReal(-1/2) # negative for outgoing fields
angularl = ArbReal(1/2)

Nc = 64 # fixes level for convergents
Nf = 32 # fixes level for Fredholm expansion
MAXSTEPS =200
																																																																																																																																																											 # fixes max no of steps for root-finding

## Compute the tau for PV using the Fredholm determinant expansion

function pieV(th,sig)
    return (ArbNumerics.gamma(1-sig)^2*ArbNumerics.rgamma(1+sig)^2 *
            ArbNumerics.gamma(1+(th[3]+sig)/2)*ArbNumerics.rgamma(1+(th[3]-sig)/2) *
            ArbNumerics.gamma(1+(th[2]+th[1]+sig)/2)*ArbNumerics.rgamma(1+(th[2]+th[1]-sig)/2) *
            ArbNumerics.gamma(1+(th[2]-th[1]+sig)/2)*ArbNumerics.rgamma(1+(th[2]-th[1]-sig)/2))
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

function gee(th1,th2,th3)
    psi = zeros(ArbComplex,2,2,Nf+1)
    # Minus sign on th2 to conform with Nekrasov expansion
    a = (th1-th2+th3)/2
    b = (th1-th2-th3)/2
    c = th1
    # psi is actually invariant by conjugation by a diagonal matrix
    psi[:,:,1] = [ 1 0 ; 0 1 ]
    psi[:,:,2] = [ (a*b/c) (-a*b/c/(1+c)) ; ((a-c)*(b-c)/c/(1-c)) (-(a-c)*(b-c)/c) ]
    for p = 3:Nf+1
        psi[1,1,p] = ((a+p-2)*(b+p-2)/((c+p-2)*(p-1))*psi[1,1,p-1])
        psi[1,2,p] = ((a+p-2)*(b+p-2)/((c+p-1)*(p-2))*psi[1,2,p-1]) ;
        psi[2,1,p] = ((a-c+p-2)*(b-c+p-2)/((-c+p-1)*(p-2))*psi[2,1,p-1])
        psi[2,2,p] = ((a-c+p-2)*(b-c+p-2)/((-c+p-2)*(p-1))*psi[2,2,p-1])
    end
    return psi
end

function geeV(th1,th2,t0=1.0)
    psi = zeros(ArbComplex,2,2,Nf+1)
    # Different definition to LNR2018
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

function BuildA(sig,th1,th2)
    vecg = gee(sig,th1,th2)
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

function BuildD(sig,ths,x,t0)
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

function tauhat(th,sig,esse,t)
    x = esse*pieV(th,sig)*t^sig
    # theta[1] = theta1, theta[2] = thetainfinity, theta[3] = thetastar
    OPA = BuildA(sig,th[2],th[1])
    OPD = BuildD(-sig,th[3],x,-t)
    id = Matrix{ArbComplex}(I,2Nf,2Nf)
    Fred = ArbNumerics.ArbComplexMatrix(id - OPA*OPD)
    return ArbNumerics.determinant(Fred)
end

function firstlogdiffV(th,sig,esse,t)
    x = esse*pieV(th,sig)*t^sig
    OpA = BuildA(sig,th[2],th[1])
    OpD = BuildD(-sig,th[3],x,-t)
    OpDdiff = BuildDdiff(-sig,th[3],x,-t)
    Id = Matrix{ArbComplex}(I,2Nf,2Nf)
    OpKi = (ArbNumerics.ArbComplexMatrix(Id-OpA*OpD))^(-1)
    OpKdiff = ArbNumerics.ArbComplexMatrix(OpA*OpDdiff)
    return ArbNumerics.tr(ArbNumerics.ArbComplexMatrix(OpKdiff*OpKi))
end

function accessoryKV(th,sig,esse,t)
    theta = copy(th)
    theta[3] -= 1
    # Additional Schlesinger move th[2]=>th[2]-1,th[1]=>th[1]-1
    # prior to shift th[1]=>th[1]+1?
    theta[2] -= 1
    sig -= 1
    # esse is invariant
    prefactor = (sig^2-theta[3]^2)/(4t)+theta[2]
    return prefactor+firstlogdiffV(theta,sig,esse,t)
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


#=
#  ADICIONADO CONSIDERANDO A MUDANÇA \THETA[3] -> -\THETA[3]

# The continuous fraction method below 

function eirad(th,sig,n)
    return (sig-th[3]+2*n-4)*(sig+th[3]-2*th[2]+2*n) +
           (2+th[3]-th[2])^2-th[1]^2
end

function beerad(th,sig,n)
    return (sig-th[3]+2*n-2)*(sig+th[3]+2*n)
end

function seerad(th,sig,n)
    return 2*(sig-th[3]+2*n-2)
end

function deerad(th,sig,n)
    return 2*(sig-th[3]+2*n)
end

function you0rad(th,sig,K,t)
    res = 0
    i = Nc
    while ( i > 0 )
        res = eirad(th,sig,i)/(beerad(th,sig,i)-4*K+t*(seerad(th,sig,i) -
              deerad(th,sig,i)*res))
        i -= 1
    end
    return res
end

function vee0rad(th,sig,K,t)
    res = 0
    i = Nc
    while ( i > 0 )
        res = deerad(th,sig,-i)/(beerad(th,sig,-i)-4*K+t*(seerad(th,sig,-i) -
              eirad(th,sig,-i)*res))
        i -= 1
    end
    return res
end

function equationrad(th,sig,K,t)
    return t*(eirad(th,sig,0)*vee0rad(th,sig,K,t)-seerad(th,sig,0) +
           deerad(th,sig,0)*you0rad(th,sig,K,t))+4*K-beerad(th,sig,0)
end

=#

function muller(f,x,verb,tol=1e-40,maxsteps=MAXSTEPS::Int64)
    h = ArbFloat(1e-8)

	local xv = [ ball(ArbComplex(x))[1] ball(ArbComplex(x+h))[1] ball(ArbComplex(x-h))[1] ]
	local fv = [ ball(ArbComplex(f(xv[1])))[1] ball(ArbComplex(f(xv[2])))[1] ball(ArbComplex(f(xv[3])))[1] ]
	local counter = 1
    #println(fv)

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

function newton2d(f,x,verb,tol=1e-30,maxsteps=MAXSTEPS::Int64)
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


# angular system
#=
function angulareq(z,q)
    thang = zeros(ArbComplex,3)
    thang[1] = -angularm-spin
    thang[2] = angularm-spin
    thang[3] = -2.0*spin
    sigang = (2.0*angularl+2.0)   # sigang -> -sigang for m<0
    tang = -4*z*ArbNumerics.cos(eta)
    extrafac = 2*(2*spin-angularm+1)*z*ArbNumerics.cos(eta)+z^2*ArbNumerics.cos(eta)^2

    return equation(thang,sigang,q+extrafac,tang)
end
=#

# radial system
function radialeq(z,qQ)
    thrad = zeros(ArbComplex,3)
    # z = M*omega
    # rplus/M = 1+sin(eta), rminus/M = 1-sin(eta)
    thrad[1] = -spin-1im*(qQ-z*(1-ArbNumerics.sin(eta)))/ArbNumerics.sin(eta)+1im*(qQ-z*(1-ArbNumerics.sin(eta)))
    thrad[2] = spin-1im*(qQ-z*(1+ArbNumerics.sin(eta)))/ArbNumerics.sin(eta)-1im*(qQ-z*(1+ArbNumerics.sin(eta)))
    thrad[3] = -2*spin+4im*z-2*im*qQ
    zrad = 4im*z*ArbNumerics.sin(eta)
    #println(zrad*thrad)
    
    lamb = (angularl-spin)*(angularl+spin+1)
    extrafac = lamb+4*qQ*z*(1+ArbNumerics.sin(eta))-4*z^2*(1+ArbNumerics.sin(eta))^2-im*((1+2*spin)*(qQ-2*z)+2*z*ArbNumerics.sin(eta)) 

    function func1(x)
        return equation(thrad,x,extrafac,zrad)
    end
    sig0 = muller( func1, ArbComplex(1.1-0.1im), false )[1]
    #println(Complex{Float64}(sig0))
    #println(Complex{Float64}(abs(4*z*(2*z-angularm)))-Complex{Float64}(abs(zrad/2*(thrad[2]-thrad[1]))))
    # this is the quantization condition
    ess0 = ArbNumerics.exp(-1im*pi*sig0)*ArbNumerics.sinpi((thrad[2]+thrad[1]+sig0)/2) *
           ArbNumerics.sinpi((thrad[2]-thrad[1]+sig0)/2)*ArbNumerics.sinpi((thrad[3]+sig0)/2) /
           ArbNumerics.sinpi((thrad[2]+thrad[1]-sig0)/2)/ArbNumerics.sinpi((thrad[2]-thrad[1]-sig0)/2)/
           ArbNumerics.sinpi((thrad[3]-sig0)/2)
    #println(Complex{Float64}(ess0))
    #println(ComplexF64((q+extrafac)/zrad))
    #println(ComplexF64(accessoryKV(thrad,sig0,ess0,zrad)))
    return tauhat(thrad,sig0,ess0,zrad)
end

function fr_wrapper(x) 
    res = radialeq(x,qQ)
    return res
end

#=
zinit = ArbComplex( 0.373671768441 - 0.08896231568im )
#zinit = ArbComplex(0.390568555724278 - 0.08011807023175106im)
zerox, count = muller(schwarzschild, zinit, true)
println(zerox)
#println(schwarzschild(zerox))
println(printschwarzschild(zerox))
=#


eta = ArbNumerics.acos(0.0)

function rang(ini)
    i = ini
    arq = open("RNQnm_Subextremal.txt","w")
    zinitial = ArbComplex(0.23818202190476453 - 0.08768487395814532im)
    while ( i < 1.01 )
        global  qQ=i   # cos(eta)=Q/M; Need this as global
	solution = muller(fr_wrapper, zinitial, false )[1]
	zrad = 4im*(solution[1][1])*ArbNumerics.sin(eta)
	thrad1 = -spin-1im*(qQ-(solution[1][1])*(1-ArbNumerics.sin(eta)))/ArbNumerics.sin(eta)+1im*(qQ-(solution[1][1])*(1-ArbNumerics.sin(eta)))
        thrad2 = spin-1im*(qQ-(solution[1][1])*(1+ArbNumerics.sin(eta)))/ArbNumerics.sin(eta)-1im*(qQ-(solution[1][1])*(1+ArbNumerics.sin(eta)))
	println("qQ=",ArbFloat(qQ,digits=8),",","\t","omeg=",Complex{Float64}(solution[1][1]),",","\t","L=",Complex{Float64}(thrad1+thrad2),",","\t","zrad=",Complex{Float64}(zrad) )
	println(arq,ArbFloat(qQ,digits=8),",","\t",Complex{Float64}(solution[1][1]))
        zinitial = solution
        i += 0.01 #increment
    end
    return close(arq)
end


#start point for the range
#rang(0.0)




qQ =0.68

function rang(ini)
    i = ini
    arq = open("RNQnm_Subextremal.txt","w")
    zinitial = ArbComplex(0.6800237348004282 - 0.002465308633427434im)
    while ( i < 0.99999999)
        global  eta=ArbNumerics.acos(i)   # cos(eta)=Q/M; Need this as global
	solution = muller(fr_wrapper, zinitial, false )[1]
	thrad1 = -spin-1im*(qQ-(solution[1][1])*(1-ArbNumerics.sin(eta)))/ArbNumerics.sin(eta)+1im*(qQ-(solution[1][1])*(1-ArbNumerics.sin(eta)))
        thrad2 = spin-1im*(qQ-(solution[1][1])*(1+ArbNumerics.sin(eta)))/ArbNumerics.sin(eta)-1im*(qQ-(solution[1][1])*(1+ArbNumerics.sin(eta)))
	println("Q/M=",ArbFloat(i,digits=8),",","\t","omeg=",Complex{Float64}(solution[1][1]),",","\t","L=",Complex{Float64}(thrad1+thrad2))
	println(arq,ArbFloat(i,digits=8),",","\t",Complex{Float64}(solution[1][1]))
        zinitial = solution
        i += 0.00000001  #increment
    end
    return close(arq)
end

#start point for the range
rang(0.999998)

#=
nu = 0.0001

# beta_1 calculation
function beta1factor(qQ)
    alph0 = ArbNumerics.sqrt((2*angularl+1)^2-4*qQ^2)
    arq = open("/home/joaocavalcante/Dropbox/Painlevé V/RN Black Hole/Subextremal RN/s=1over2, l=1,3,5,7over2/RN_beta1_extremal.txt","w")
        
    function func1(x)
        return  ArbNumerics.exp(-1im*pi*alpha0/2)*(ArbNumerics.gamma(1-alph0)^2*ArbNumerics.rgamma(1+alph0)^2 *
                ArbNumerics.gamma((1+alph0)/2-im*x)*ArbNumerics.rgamma((1-alph0)/2-im*x) *
                ArbNumerics.gamma((1+alph0)/2-im*qQ-s)*ArbNumerics.rgamma((1-alph0)/2-im*qQ-s) *
                ArbNumerics.gamma((1+alph0)/2-im*qQ+s)*ArbNumerics.rgamma((1-alph0)/2-im*qQ+s)*(4*nu*qQ)^alph0-1
    end
    
    qQ=i
    betainit = ArbComplex(0.6600871042203367 - 0.03946864525671397im)
    while ( i < 0.99999)
        global  qQ = i 
        solution = muller( func1, ArbComplex(1.1-0.1im), false )[1]
        println("qQ=",ArbFloat(i,digits=8),",","\t","beta=",Complex{Float64}(solution[1][1]))
        println(arq,ArbFloat(i,digits=8),",","\t",Complex{Float64}(solution[1][1]))
        betainit = solution
        i += 0.0000005  #increment
    end
    return close(arq)
end
qQ= 0.

=#

#omega_Berti
#global eta = ArbNumerics.acos(0.99)
#zberti = ArbComplex(0.2921110527 - 0.08805736756im)
#zinit = ArbComplex(0.2911543120662345 - 0.08596666491983046im)
#qberti = ArbComplex(4.716300076 - 0.1966553372im)
#q = ArbComplex(4.71391995002163 - 0.192098275800669im)
#println("radial_berti =",Complex{Float64}(radialeq(zberti,qberti)))
#println("radial_our =",Complex{Float64}(radialeq(zinit,q)))
#println("angular_berti=",Complex{Float64}(angulareq(zberti,qberti)))
#println("angular_our=",Complex{Float64}(angulareq(zinit,q)))

#=
global eta = ArbNumerics.acos(0.99) # cos(eta)=a/M; Need this as global
initial = zeros(ArbComplex,2)
# Berti's values for a/M = 0.8
initial[1] = ArbComplex(0.1104468641 - 0.08949886198im)
initial[2] = ArbComplex(-0.001363056469 + 0.006461127160im)
#println(angulareq(initial[1],initial[2]))
#println(radialeq(initial[1],initial[2]))
solution = newton2d(fr_wrapper, initial, true)
println("z = ",solution[1][1])
println("l = ",solution[1][2])
println("Verify_z= ", abs2d(fr_wrapper(solution[1])))
=#

