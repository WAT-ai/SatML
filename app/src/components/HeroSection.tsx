'use client';

import { Globe } from "@/components/magicui/globe";
import Link from 'next/link';
import { AuroraText } from "@/components/magicui/aurora-text";

export default function HeroSection() {
  return (
    <section className="relative w-full h-[85vh] bg-black text-primary overflow-hidden">
      <div className="relative flex flex-row items-center justify-between max-w-7xl mx-auto h-full px-10">
        
        {/* Image on the left */}
        <div className="w-1/2 h-full relative">
          <Globe />
        </div>

        {/* Text on the right */}
        <div className="w-1/2 flex flex-col justify-center items-start px-8">
          <h1 className="text-3xl font-normal max-w-4xl leading-tight text-white pb-2">
          <AuroraText>AI-powered</AuroraText> methane emissions detection via hyper-spectral satellite imagery.
          </h1>
          <p className="text-gray-400 text-lg font-extralight italic pb-4">
            Analyze. Predict. Prevent.
          </p>

          <Link href="#" className="mt-4 px-6 py-2 border border-white text-white rounded-md hover:bg-white hover:text-black transition">
            Learn More  →
          </Link>
        </div>

      </div>
    </section>
  );
}
