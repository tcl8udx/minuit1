// Using Minuit to fit a double gaussian and a gobel distribution to some data

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>

#include <TMinuit.h>
#include <TApplication.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TROOT.h>
#include <TMath.h>
#include <TH1F.h>
#include <TF1.h>
#include <TString.h>
#include <TAxis.h>
#include <TLine.h>
#include <TFile.h>
#include <TLegend.h>
#include <TPaveText.h>

using namespace std;

// ---------- GLOBALS ----------
TH1F *hdata;
TF1 *fparam;

//-------------------------------------------------------------------------
// Two Gaussian PDF
double twoGaussian(double* xPtr, double par[]){        
  double x = *xPtr;
  double A1 = par[0];      // amplitude of first Gaussian
  double mu1 = par[1];     // mean of first Gaussian
  double sigma1 = par[2];  // width of first Gaussian
  double A2 = par[3];      // amplitude of second Gaussian
  double mu2 = par[4];     // mean of second Gaussian
  double sigma2 = par[5];  // width of second Gaussian
  
  double gauss1 = A1 * TMath::Gaus(x, mu1, sigma1, false);
  double gauss2 = A2 * TMath::Gaus(x, mu2, sigma2, false);
  
  return gauss1 + gauss2;
}

//-------------------------------------------------------------------------
// Gumbel Distribution PDF
double gumbelPdf(double* xPtr, double par[]){        
  double x = *xPtr;
  double A = par[0];       // normalization/amplitude
  double mu = par[1];      // location parameter
  double beta = par[2];    // scale parameter
  
  if (beta <= 0) return 0;
  
  double z = (x - mu) / beta;
  double f = A * (1.0/beta) * TMath::Exp(-(z + TMath::Exp(-z)));
  
  return f;
}

//-------------------------------------------------------------------------
// Calculate Chi-square
double calcCHI(TH1F* h, TF1* f){
  double chisq = 0;
  int ndf = 0;
  
  for (int i=1; i<=h->GetNbinsX(); i++){
    double x = h->GetBinCenter(i);
    double expected = f->Eval(x);
    double observed = h->GetBinContent(i);
    double err = h->GetBinError(i);
    
    if (err < 1e-10) err = 1e-10;  // avoid division by zero
    if (observed > 0) {  // only include bins with data
      chisq += (observed - expected)*(observed - expected)/(err*err);
      ndf++;
    }
  }
  
  return chisq;
}

//-------------------------------------------------------------------------
// Calculate NLL
double calcNLL(TH1F* h, TF1* f){
  double nll = 0;
  
  for (int i=1; i<=h->GetNbinsX(); i++){
    double x = h->GetBinCenter(i);
    int n = (int)(h->GetBinContent(i));
    double mu = f->Eval(x);
    
    if (mu < 1e-10) mu = 1e-10;
    if (n > 0) {
      nll -= n * TMath::Log(mu) - mu - TMath::LnGamma(n+1);
    } else {
      nll -= -mu;
    }
  }
  
  return 2*nll;
}

//-------------------------------------------------------------------------
// Minuit objective function for chi-square
void fcn_chi(int& npar, double* deriv, double& f, double par[], int flag){
  for (int i=0; i<npar; i++){
    fparam->SetParameter(i, par[i]);
  }
  
  f = calcCHI(hdata, fparam);
}

//-------------------------------------------------------------------------
// Minuit objective function for NLL
void fcn_nll(int& npar, double* deriv, double& f, double par[], int flag){
  for (int i=0; i<npar; i++){
    fparam->SetParameter(i, par[i]);
  }
  
  f = calcNLL(hdata, fparam);
}

//-------------------------------------------------------------------------
// Fit function
void fitHistogram(TH1F* hist, TF1* func, int npar, double* initialPar, 
                  double* stepSize, double* minVal, double* maxVal, 
                  TString* parName, double& chi2, int& ndf, double& pvalue,
                  bool useNLL = false) {
  
  hdata = hist;
  fparam = func;
  
  TMinuit minuit(npar);
  if (useNLL) {
    minuit.SetFCN(fcn_nll);
  } else {
    minuit.SetFCN(fcn_chi);
  }
  minuit.SetPrintLevel(-1);  // suppress output
  
  // Initialize parameters
  for (int i=0; i<npar; i++){
    minuit.DefineParameter(i, parName[i].Data(), 
                          initialPar[i], stepSize[i], minVal[i], maxVal[i]);
  }
  
  // Do the minimization
  minuit.Migrad();
  
  // Get results
  double outpar[npar], err[npar];
  for (int i=0; i<npar; i++){
    minuit.GetParameter(i, outpar[i], err[i]);
  }
  
  func->SetParameters(outpar);
  
  // Calculate chi-square and NLL
  chi2 = calcCHI(hist, func);
  double nll = calcNLL(hist, func);
  
  ndf = 0;
  for (int i=1; i<=hist->GetNbinsX(); i++){
    if (hist->GetBinContent(i) > 0) ndf++;
  }
  ndf -= npar;
  
  if (useNLL) {
    pvalue = TMath::Prob(nll, ndf);
  } else {
    pvalue = TMath::Prob(chi2, ndf);
  }
  
  // Print results
  cout << "\nFit Results for " << func->GetName();
  cout << (useNLL ? " (NLL)" : " (Chi-square)") << ":" << endl;
  cout << "Chi-square = " << chi2 << endl;
  cout << "NLL = " << nll << endl;
  cout << "NDF = " << ndf << endl;
  cout << "Chi-square/NDF = " << chi2/ndf << endl;
  cout << "NLL/NDF = " << nll/ndf << endl;
  cout << "P-value = " << pvalue;
  cout << (useNLL ? " (from NLL)" : " (from Chi-square)") << endl;
  cout << "Parameters:" << endl;
  for (int i=0; i<npar; i++){
    cout << "  " << parName[i] << " = " << outpar[i] << " +/- " << err[i] << endl;
  }
}

//-------------------------------------------------------------------------

int main(int argc, char **argv) {

  TApplication theApp("App", &argc, argv);

  // Set style
  gStyle->SetOptStat(0);
  gStyle->SetTitleBorderSize(0);
  gStyle->SetTitleSize(0.04);
  gStyle->SetTitleFont(42, "hxy");
  gStyle->SetLabelFont(42, "xyz");
  gROOT->ForceStyle();

  // Open ROOT file
  TFile* file = TFile::Open("distros.root");
  if (!file || file->IsZombie()) {
    cout << "Error: Cannot open distros.root" << endl;
    return 1;
  }

  // Get histogram dist1
  TH1F* dist1 = (TH1F*)file->Get("dist1");
  if (!dist1) {
    cout << "Error: Cannot find dist1 histogram" << endl;
    return 1;
  }

  // Create canvas with two pads
  TCanvas* canvas = new TCanvas("canvas", "Distribution Fits", 1200, 600);
  canvas->Divide(2, 1);

  // Get histogram range
  double xmin = dist1->GetXaxis()->GetXmin();
  double xmax = dist1->GetXaxis()->GetXmax();

  cout << "\n========================================" << endl;
  cout << "Fitting dist1 histogram with Chi-square" << endl;
  cout << "========================================" << endl;

  // ========== FIT 1: Two Gaussians (Chi-square) ==========
  const int npar1 = 6;
  TF1* fit2gauss = new TF1("fit2gauss", twoGaussian, xmin, xmax, npar1);
  
  double par1[npar1];
  double stepSize1[npar1];
  double minVal1[npar1];
  double maxVal1[npar1];
  TString parName1[npar1];
  
  // Initial guesses based on histogram
  double maxVal_hist = dist1->GetMaximum();
  double meanVal = dist1->GetMean();
  double rmsVal = dist1->GetRMS();
  
  // Better initial guesses for Gumbel-like data
  // Main peak
  par1[0] = maxVal_hist * 0.7;     // A1 - main Gaussian
  par1[1] = dist1->GetBinCenter(dist1->GetMaximumBin());  // mu1 - at peak
  par1[2] = rmsVal * 0.3;          // sigma1 - narrow
  // Tail
  par1[3] = maxVal_hist * 0.3;     // A2 - tail Gaussian
  par1[4] = meanVal + rmsVal * 0.5; // mu2 - shifted right
  par1[5] = rmsVal * 1.5;          // sigma2 - broader for tail
  
  for (int i=0; i<npar1; i++){
    stepSize1[i] = TMath::Abs(par1[i] * 0.1);
    minVal1[i] = 0;
    maxVal1[i] = 0;
  }
  
  parName1[0] = "A1";
  parName1[1] = "mu1";
  parName1[2] = "sigma1";
  parName1[3] = "A2";
  parName1[4] = "mu2";
  parName1[5] = "sigma2";
  
  double chi2_2g, pval_2g;
  int ndf_2g;
  fitHistogram(dist1, fit2gauss, npar1, par1, stepSize1, minVal1, maxVal1, 
               parName1, chi2_2g, ndf_2g, pval_2g);
  
  fit2gauss->SetLineColor(kRed);
  fit2gauss->SetLineWidth(2);

  // ========== FIT 2: Gumbel Distribution ==========
  const int npar2 = 3;
  TF1* fitGumbel = new TF1("fitGumbel", gumbelPdf, xmin, xmax, npar2);
  
  double par2[npar2];
  double stepSize2[npar2];
  double minVal2[npar2];
  double maxVal2[npar2];
  TString parName2[npar2];
  
  par2[0] = maxVal_hist * 2;    // A (normalization)
  par2[1] = meanVal;             // mu (location)
  par2[2] = rmsVal;              // beta (scale)
  
  for (int i=0; i<npar2; i++){
    stepSize2[i] = TMath::Abs(par2[i] * 0.1);
    minVal2[i] = 0;
    maxVal2[i] = 0;
  }
  
  parName2[0] = "A";
  parName2[1] = "mu";
  parName2[2] = "beta";
  
  double chi2_gumbel, pval_gumbel;
  int ndf_gumbel;
  fitHistogram(dist1, fitGumbel, npar2, par2, stepSize2, minVal2, maxVal2, 
               parName2, chi2_gumbel, ndf_gumbel, pval_gumbel);
  
  fitGumbel->SetLineColor(kBlue);
  fitGumbel->SetLineWidth(2);

  // ========== PLOTTING ==========
  
  // Plot 1: Two Gaussians
  canvas->cd(1);
  gPad->SetLeftMargin(0.12);
  gPad->SetRightMargin(0.05);
  
  dist1->SetLineColor(kBlack);
  dist1->SetMarkerStyle(20);
  dist1->SetMarkerSize(0.8);
  dist1->GetXaxis()->SetTitle("x");
  dist1->GetYaxis()->SetTitle("Events");
  dist1->SetTitle("Two Gaussians Fit");
  dist1->Draw("E");
  fit2gauss->Draw("same");
  
  TLegend* leg1 = new TLegend(0.55, 0.65, 0.92, 0.88);
  leg1->SetBorderSize(0);
  leg1->SetFillStyle(0);
  leg1->AddEntry(dist1, "Data", "lep");
  leg1->AddEntry(fit2gauss, "Two Gaussians", "l");
  leg1->AddEntry((TObject*)0, Form("#chi^{2} = %.2f", chi2_2g), "");
  leg1->AddEntry((TObject*)0, Form("p-value = %.4f", pval_2g), "");
  leg1->Draw();
  
  // Plot 2: Gumbel
  canvas->cd(2);
  gPad->SetLeftMargin(0.12);
  gPad->SetRightMargin(0.05);
  
  TH1F* dist1_copy = (TH1F*)dist1->Clone("dist1_copy");
  dist1_copy->SetTitle("Gumbel Distribution Fit");
  dist1_copy->Draw("E");
  fitGumbel->Draw("same");
  
  TLegend* leg2 = new TLegend(0.55, 0.65, 0.92, 0.88);
  leg2->SetBorderSize(0);
  leg2->SetFillStyle(0);
  leg2->AddEntry(dist1_copy, "Data", "lep");
  leg2->AddEntry(fitGumbel, "Gumbel", "l");
  leg2->AddEntry((TObject*)0, Form("#chi^{2} = %.2f", chi2_gumbel), "");
  leg2->AddEntry((TObject*)0, Form("p-value = %.4f", pval_gumbel), "");
  leg2->Draw();

  // Summary
  cout << "COMPARISON SUMMARY" << endl;
  
  cout << "\n*** Chi-square Fits ***" << endl;
  cout << "\nTwo Gaussians:" << endl;
  cout << "  Chi^2/NDF = " << chi2_2g << "/" << ndf_2g 
       << " = " << chi2_2g/ndf_2g << endl;
  cout << "  P-value = " << pval_2g << endl;
  
  cout << "\nGumbel Distribution:" << endl;
  cout << "  Chi^2 = " << chi2_gumbel << "/" << ndf_gumbel 
       << " = " << chi2_gumbel/ndf_gumbel << endl;
  cout << "  P-value = " << pval_gumbel << endl;
  
  canvas->Update();
  canvas->SaveAs("ex1.pdf(");
  


  // ========================================
  // NOW DO NLL-BASED FITS
  // ========================================
  
  cout << "\n========================================" << endl;
  cout << "Fitting dist1 histogram with NLL" << endl;
  cout << "========================================" << endl;
  
  TCanvas* canvas2 = new TCanvas("canvas2", "Distribution Fits (NLL)", 1200, 600);
  canvas2->Divide(2, 1);
  
  // ========== FIT 1: Two Gaussians (NLL) ==========
  TF1* fit2gauss_nll = new TF1("fit2gauss_nll", twoGaussian, xmin, xmax, npar1);
  
  // Use same initial parameters
  double chi2_2g_nll, pval_2g_nll;
  int ndf_2g_nll;
  fitHistogram(dist1, fit2gauss_nll, npar1, par1, stepSize1, minVal1, maxVal1, 
               parName1, chi2_2g_nll, ndf_2g_nll, pval_2g_nll, true);  // true = use NLL
  
  fit2gauss_nll->SetLineColor(kRed);
  fit2gauss_nll->SetLineWidth(2);
  
  // ========== FIT 2: Gumbel Distribution (NLL) ==========
  TF1* fitGumbel_nll = new TF1("fitGumbel_nll", gumbelPdf, xmin, xmax, npar2);
  
  double chi2_gumbel_nll, pval_gumbel_nll;
  int ndf_gumbel_nll;
  fitHistogram(dist1, fitGumbel_nll, npar2, par2, stepSize2, minVal2, maxVal2, 
               parName2, chi2_gumbel_nll, ndf_gumbel_nll, pval_gumbel_nll, true);  // true = use NLL
  
  fitGumbel_nll->SetLineColor(kBlue);
  fitGumbel_nll->SetLineWidth(2);

  // Summary
  cout << "COMPARISON SUMMARY" << endl;

  cout << "\nTwo Gaussians:" << endl;
  cout << "  NLL/NDF = " << calcNLL(dist1, fit2gauss_nll) << "/" << ndf_2g_nll 
       << " = " << calcNLL(dist1, fit2gauss_nll)/ndf_2g_nll << endl;
  cout << "  P-value = " << pval_2g_nll << endl;
  
  cout << "\nGumbel Distribution:" << endl;
  cout << "  NLL/NDF = " << calcNLL(dist1, fitGumbel_nll) << "/" << ndf_gumbel_nll 
       << " = " << calcNLL(dist1, fitGumbel_nll)/ndf_gumbel_nll << endl;
  cout << "  P-value = " << pval_gumbel_nll << endl;
  
  cout << "\nConclusion (NLL):" << endl;
  if (pval_2g_nll > pval_gumbel_nll) {
    cout << "  Two Gaussians fit is preferred (higher p-value)" << endl;
  } else {
    cout << "  Gumbel fit is preferred (higher p-value)" << endl;
  }
  
  // ========== PLOTTING (NLL) ==========
  
  // Plot 1: Two Gaussians (NLL)
  canvas2->cd(1);
  gPad->SetLeftMargin(0.12);
  gPad->SetRightMargin(0.05);
  
  TH1F* dist1_nll = (TH1F*)dist1->Clone("dist1_nll");
  dist1_nll->SetTitle("Two Gaussians Fit");
  dist1_nll->Draw("E");
  fit2gauss_nll->Draw("same");
  
  TLegend* leg1_nll = new TLegend(0.55, 0.65, 0.92, 0.88);
  leg1_nll->SetBorderSize(0);
  leg1_nll->SetFillStyle(0);
  leg1_nll->AddEntry(dist1_nll, "Data", "lep");
  leg1_nll->AddEntry(fit2gauss_nll, "Two Gaussians", "l");
  leg1_nll->AddEntry((TObject*)0, Form("NLL = %.2f", calcNLL(dist1, fit2gauss_nll), ""));
  // leg1_nll->AddEntry((TObject*)0, Form("p-value = %.4f", pval_2g_nll), "");
  leg1_nll->Draw();
  
  // Plot 2: Gumbel (NLL)
  canvas2->cd(2);
  gPad->SetLeftMargin(0.12);
  gPad->SetRightMargin(0.05);
  
  TH1F* dist1_nll2 = (TH1F*)dist1->Clone("dist1_nll2");
  dist1_nll2->SetTitle("Gumbel Distribution");
  dist1_nll2->Draw("E");
  fitGumbel_nll->Draw("same");
  
  TLegend* leg2_nll = new TLegend(0.55, 0.65, 0.92, 0.88);
  leg2_nll->SetBorderSize(0);
  leg2_nll->SetFillStyle(0);
  leg2_nll->AddEntry(dist1_nll2, "Data", "lep");
  leg2_nll->AddEntry(fitGumbel_nll, "Gumbel", "l");
  leg2_nll->AddEntry((TObject*)0, Form("NLL = %.2f", calcNLL(dist1, fitGumbel_nll), ""));
  // leg2_nll->AddEntry((TObject*)0, Form("p-value = %.4f", pval_gumbel_nll), "");
  leg2_nll->Draw();
  
  canvas2->Update();
  canvas2->SaveAs("ex1.pdf)");  // Close the PDF
    
  theApp.SetIdleTimer(1, ".q");
  theApp.Run(true);
  
  canvas->Close();
  canvas2->Close();
  file->Close();

  return 0;
}