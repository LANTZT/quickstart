/* -*- mode: c++; coding: utf-8; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4; show-trailing-whitespace: t  -*- vim:set fenc=utf-8:ft=tcl:et:sw=4:ts=4:sts=4

   This file is part of the Feel++ library

   Author(s): Christophe Prud'homme <christophe.prudhomme@feelpp.org>
Date     : Tue Feb 25 11:58:30 2014

Copyright (C) 2014 Feel++ Consortium

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/
//! [global]
#include <feel/feel.hpp>
//! [include]
#include <feel/feelmodels/modelproperties.hpp>
//! [include]

int main(int argc, char**argv )
{
  //! [env]
  using namespace Feel;
  po::options_description laplacianoptions( "Laplacian options" );
  laplacianoptions.add_options()
    ("myModel", po::value< std::string >()-> default_value("model"), "Name of our model")
    ("myVerbose", po::value< bool >()-> default_value( true ), "Display information during execution")
    ;

  Environment env( _argc=argc, _argv=argv,
      _desc=laplacianoptions,
      _about=about(_name="aniso_laplacian",
        _author="Feel++ Consortium",
        _email="feelpp-devel@feelpp.org"));
  //! [env]

  //! [load_model]
  ModelProperties model(Environment::expand(soption("myModel")));
  //! [load_model]
  //! [get_bc]
  map_scalar_field<2> bc_u { model.boundaryConditions().getScalarFields<2>("heat","dirichlet") };
  //! [get_bc]
  //! [get_mat]
  ModelMaterials materials = model.materials();
  //! [get_mat]

  if(boption("myVerbose") && Environment::isMasterRank() )
    std::cout << "Model " << Environment::expand( soption("myModel")) << " loaded." << std::endl;

  //! [rhs]
  auto f = expr( soption(_name="functions.f"), "f" );
  //! [rhs]

  //! [function_space]
  auto mesh = loadMesh(_mesh=new Mesh<Simplex<MODEL_DIM>>);
  auto Vh = Pch<2>( mesh );
  auto u = Vh->element();
  auto v = Vh->element();
  auto k11 = Vh->element();
  auto k12 = Vh->element();
  auto k22 = Vh->element();
#if MODEL_DIM == 3
  auto k13 = Vh->element();
  auto k23 = Vh->element();
  auto k33 = Vh->element();
#endif
  //! [function_space]

  //! [forms]
  auto a = form2( _trial=Vh, _test=Vh);
  auto l = form1( _test=Vh );
  //! [forms]

  //! [applying_rhs]
  l = integrate(_range=elements(mesh),_expr=f*id(v));
  //! [applying_rhs]

  //! [materials]
  for(auto it : materials)
  {
    auto mat = material(it);
    if(boption("myVerbose") && Environment::isMasterRank() )
      std::cout << "[Materials] - Laoding data for " << it.second.name() << " that apply on marker " << it.first  << " with diffusion coef [" 
#if MODEL_DIM == 3
        << "[" << it.second.k11() << "," << it.second.k12() << "," << it.second.k13() << "],"
        << "[" << it.second.k12() << "," << it.second.k22() << "," << it.second.k23() << "],"
        << "[" << it.second.k13() << "," << it.second.k23() << "," << it.second.k33() << "]]" 
#else
        << "[" << it.second.k11() << "," << it.second.k12() << "],"
        << "[" << it.second.k12() << "," << it.second.k22() << "]]"
#endif
        << std::endl;
    k11.on(_range=markedelements(mesh,it.first),_expr=cst(it.second.k11()));
    k12.on(_range=markedelements(mesh,it.first),_expr=cst(it.second.k12()));
    k22.on(_range=markedelements(mesh,it.first),_expr=cst(it.second.k22()));
#if MODEL_DIM == 3
    k13 += vf::project(_space=Vh,_range=markedelements(mesh,marker(it)),_expr=mat.k13());
    k23 += vf::project(_space=Vh,_range=markedelements(mesh,marker(it)),_expr=mat.k23());
    k33 += vf::project(_space=Vh,_range=markedelements(mesh,marker(it)),_expr=mat.k33());
#endif
  }
#if MODEL_DIM == 2
  a += integrate(_range=elements(mesh),_expr=inner(mat<MODEL_DIM,MODEL_DIM>(idv(k11), idv(k12), idv(k12), idv(k22) )*trans(gradt(u)),trans(grad(v))) );
#else
  a += integrate(_range=elements(mesh),_expr=inner(mat<MODEL_DIM,MODEL_DIM>(idv(k11), idv(k12), idv(k13), idv(k12), idv(k22), idv(k23), idv(k31), idv(k32), idv(k33))*trans(gradt(u)),trans(grad(v))) );
#endif
  //! [materials]

  //! [boundary]
  for(auto it : bc_u){
    if(boption("myVerbose") && Environment::isMasterRank() )
      std::cout << "[BC] - Applying " << it.second << " on " << it.first << std::endl;
    a+=on(_range=markedfaces(mesh,it.first), _rhs=l, _element=u, _expr=it.second );
  }
  //! [boundary]

  //! [solve]
  a.solve(_rhs=l,_solution=u);
  //! [solve]

  //! [export]
  auto e = exporter( _mesh=mesh );
  for(int i = 0; i < 3; i ++){
    for(auto const &it : model.postProcess()["Fields"] )
    {
      if(it == "diffused") 
        e->step(i)->add("diffused",u);
      else if(it == "k11")
        e->step(i)->add("k11",k11);
      else if(it == "k12")
        e->step(i)->add("k12",k12);
      else if(it == "k11")
        e->step(i)->add("k22",k22);
#if MODEL_DIM == 3
      else if(it == "k13")
        e->step(i)->add("k13",k13);
      else if(it == "k11")
        e->step(i)->add("k23",k23);
      else if(it == "k33")
        e->step(i)->add("k33",k33);
#endif
    }
    e->save();
  }
  //! [export]
  return 0;
}
//! [global]
