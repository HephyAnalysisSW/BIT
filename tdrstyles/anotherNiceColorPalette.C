
#include "TColor.h"
#include "TStyle.h"

void vanotherNiceColorPalette( Int_t NCont = 255 ) {
 const Int_t NRGBs = 3;
 Double_t stops[NRGBs] = { 0.00, 0.83, 1.00 };
 Double_t red[NRGBs]   = { 0.87, 1.00, 0.51 };
 Double_t green[NRGBs] = { 1.00, 0.20, 0.00 };
 Double_t blue[NRGBs]  = { 0.12, 0.00, 0.00 };
 TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
 gStyle->SetNumberContours(NCont);
}

void anotherNiceColorPalette( Int_t NCont = 255 ) {
 const Int_t NRGBs = 4;
 Double_t stops[NRGBs] = { 0.00, 0.25, 0.84, 1.00 };
 Double_t red[NRGBs]   = { 0.00, 0.87, 1.00, 0.51 };
 Double_t green[NRGBs] = { 0.50, 1.00, 0.20, 0.00 };
 Double_t blue[NRGBs]  = { 0.20, 0.12, 0.00, 0.00 };
 TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
 gStyle->SetNumberContours(NCont);
}


void niceColorPalette( Int_t NCont = 255 ) {
 const Int_t NRGBs = 5;
 Double_t stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
 Double_t red[NRGBs]   = { 0.00, 0.00, 0.87, 1.00, 0.51 };
 Double_t green[NRGBs] = { 0.00, 0.81, 1.00, 0.20, 0.00 };
 Double_t blue[NRGBs]  = { 0.51, 1.00, 0.12, 0.00, 0.00 };
 TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
 gStyle->SetNumberContours(NCont);
}



