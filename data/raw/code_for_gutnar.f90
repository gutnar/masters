call calc_gal_spin_vec(gal(i)%ra,gal(i)%dec,gal(i)%pos,inc,v1,v2) ! valjastab kaks galaktika telje vektorit
v1=get_rotated_gama_coordinates_crd(v1,gal(i)%gama) ! teisendab samasse systeemi, kus on filamendid
v2=get_rotated_gama_coordinates_crd(v2,gal(i)%gama) ! gama on gama valja ID... see on input kataloogis
dum=abs(dot_product(v1, gal(i)%fil_e3 )) ! arvutab cosi kahe vektori vahel

subroutine calc_gal_spin_vec(ra,dec,pos,inc,dvec1,dvec2)
    implicit none
    real(rk),intent(in):: ra,dec,pos,inc ! input values
    real(rk),dimension(1:3),intent(out):: dvec1,dvec2 ! unit spin vector
    real(rk):: inc_buf, pos_buf, ra_buf, dec_buf, u,v,w
    real(rk):: lam,eta,x,y,z,dist_buf,len
    integer:: k
	real(rk):: rpi
	!
	rpi=pi/180.0_rk
	!
    !! radians = degrees * M_PI / 180.0;
	inc_buf = inc * PI / 180.0
	pos_buf = pos * PI / 180.0
	if (pos_buf<0.0_rk) pos_buf=PI + pos_buf
	ra_buf = ra * PI / 180.0
	dec_buf = dec * PI / 180.0
    !
	!! find the componets of unit spin vectors in the local topocentric coordinate frame
	u = -sin(inc_buf) * cos(pos_buf);
	v = sin(inc_buf) * sin(pos_buf);
	w = cos(inc_buf);	! pointing away from Earth
    !
	!! compute two alternate endpoints if the vectors were shifted to the origin of the GEI frame
	!! topocentric elevation component pointing away from Earth
	dvec1(1) = -u * sin(ra_buf) - v * sin(dec_buf) * cos(ra_buf) + w * cos(dec_buf) * cos(ra_buf);
	dvec1(2) = u * cos(ra_buf) -  v * sin(dec_buf) * sin(ra_buf) + w * cos(dec_buf) * sin(ra_buf);
	dvec1(3) = v * cos(dec_buf) + w * sin(dec_buf);
	!! topocentric elevation component pointing towards Earth
	dvec2(1) = -u * sin(ra_buf) -  v * sin(dec_buf) * cos(ra_buf) - w * cos(dec_buf) * cos(ra_buf);
	dvec2(2) =  u * cos(ra_buf) -  v * sin(dec_buf) * sin(ra_buf) - w * cos(dec_buf) * sin(ra_buf);
	dvec2(3) =  v * cos(dec_buf) - w * sin(dec_buf);
	!
	! safety procedure...
	dvec1=dvec1/sqrt(sum(dvec1**2))
	dvec2=dvec2/sqrt(sum(dvec2**2))
end subroutine calc_gal_spin_vec

function get_rotated_gama_coordinates_crd(crd0,gama) result(res)
	implicit none
	real(rk),dimension(1:3),intent(in):: crd0
	integer,intent(in):: gama
	real(rk),dimension(1:3):: los,crd,crd2,ex,ey
	real(rk),dimension(1:3),save:: los1,ex1,ey1, los2,ex2,ey2, los3,ex3,ey3
	real(rk),dimension(1:3):: res
	real(rk):: ra0,dec0,dum
	logical,dimension(1:3),save:: tt=.true.
	integer:: ii
	!
	select case (gama)
	case (9) !-----------------------
	ra0=135.0; dec0=0.5_rk; ii=1
	case (12) !-----------------------
	ra0=180.0; dec0=-0.5_rk; ii=2
	case (15) !-----------------------
	ra0=217.5; dec0=0.5_rk; ii=3
	end select
	if (tt(ii)) then
		crd=get_crd(ra0,dec0,1.0_rk)
		los=crd/sqrt( sum( crd**2 ) ) ! line of sight vector
		crd=get_crd(ra0,dec0-1.0_rk,100.0_rk)
		crd2=get_crd(ra0,dec0+1.0_rk,100.0_rk)
		crd=crd2-crd
		dum=dot_product(crd,los) ! crd projektsioon los suunal
		crd2=dum*los
		crd=crd-crd2
		ex=crd/sqrt( sum( crd**2 ) ) ! see on vaatekiirega risti olev vektor
		ey(1)=los(2)*ex(3)-los(3)*ex(2)
		ey(2)=los(3)*ex(1)-los(1)*ex(3)
		ey(3)=los(1)*ex(2)-los(2)*ex(1)
		ey=ey/sqrt(sum(ey**2)) ! normalise
	end if
	if (ii==1 .and. tt(ii)) then
		los1=los; ex1=ex; ey1=ey
		tt(ii)=.false.
	end if
	if (ii==2 .and. tt(ii)) then
		los2=los; ex2=ex; ey2=ey
		tt(ii)=.false.
	end if
	if (ii==3 .and. tt(ii)) then
		los3=los; ex3=ex; ey3=ey
		tt(ii)=.false.
	end if
	!---------------------------------------
	! risti vektorid on los, ey,ey
	! projekteerime koordinaadid ymber
	res(1)=dot_product(crd0,ex)
	res(2)=dot_product(crd0,ey)
	res(3)=dot_product(crd0,los)
end function get_rotated_gama_coordinates_crd

function get_crd(ra,dec,dist) result(crd)
	implicit none
	real(rk),intent(in):: ra,dec,dist
	real(rk),dimension(1:3):: crd
	real(rk):: rpi
	rpi=pi/180.0_rk
	crd(1)=dist * cos(ra*rpi) * cos(dec*rpi)
    crd(2)=dist * sin(ra*rpi) * cos(dec*rpi)
    crd(3)=dist * sin(dec*rpi)
end function get_crd
